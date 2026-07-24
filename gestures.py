"""Face-gesture recognition and macOS action helpers for Moodito."""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GestureSpec:
    key: str
    face_gesture: str
    action: str
    shortcut: str


GESTURE_SPECS = (
    GestureSpec("double_blink", "😉 Double blink", "Applications List", "Apps"),
    GestureSpec("long_blink", "😌 Long blink", "Show Desktop", "F11 / Show Desktop"),
    GestureSpec(
        "tilt_left",
        "↙️ Tilt head left",
        "Previous Desktop / Space",
        "Ctrl + ←",
    ),
    GestureSpec(
        "tilt_right",
        "↘️ Tilt head right",
        "Next Desktop / Space",
        "Ctrl + →",
    ),
    GestureSpec("tilt_up", "⬆️ Tilt head up", "Mission Control", "Ctrl + ↑"),
    GestureSpec("tilt_down", "⬇️ Tilt head down", "App Exposé", "Ctrl + ↓"),
)
GESTURE_KEYS = tuple(spec.key for spec in GESTURE_SPECS)
DEFAULT_GESTURES = dict.fromkeys(GESTURE_KEYS, True)
DEFAULT_REQUIRE_COMMAND = True
SYSTEM_EVENT_GESTURES = (
    "long_blink",
    "tilt_left",
    "tilt_right",
    "tilt_up",
    "tilt_down",
)

_BLINK_THRESHOLD = 0.55
_BLINK_MIN_SECONDS = 0.06
_DOUBLE_BLINK_WINDOW = 0.8
_LONG_BLINK_SECONDS = 0.8
_ROLL_HOLD_SECONDS = 0.55
_PITCH_HOLD_SECONDS = 1.0
_PITCH_THRESHOLD_DEGREES = 12.0
_ROLL_THRESHOLD_DEGREES = 13.0
_POSE_RELEASE_RATIO = 0.5
_GESTURE_COOLDOWN_SECONDS = 0.75


def pose_angles_from_matrix(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float] | None:
    """Return MediaPipe face pose as ``(nose-up pitch, roll)`` in degrees."""
    try:
        matrix_00 = float(matrix[0][0])
        matrix_10 = float(matrix[1][0])
        matrix_11 = float(matrix[1][1])
        matrix_12 = float(matrix[1][2])
        matrix_21 = float(matrix[2][1])
        matrix_22 = float(matrix[2][2])
    except (IndexError, TypeError, ValueError):
        return None

    horizontal = math.hypot(matrix_00, matrix_10)
    if horizontal > 1.0e-6:
        matrix_pitch = math.atan2(matrix_21, matrix_22)
    else:
        matrix_pitch = math.atan2(-matrix_12, matrix_11)
    roll = math.atan2(matrix_10, matrix_00)
    return -math.degrees(matrix_pitch), math.degrees(roll)


def command_key_down() -> bool:
    """Return whether the Command modifier is currently held."""
    try:
        from AppKit import NSEvent, NSEventModifierFlagCommand

        return bool(NSEvent.modifierFlags() & NSEventModifierFlagCommand)
    except Exception:  # noqa: BLE001 - modifier polling must stay best-effort
        return False


def accessibility_access_granted() -> bool:
    """Return whether macOS allows Moodito to control UI through System Events."""
    try:
        framework = ctypes.CDLL(
            "/System/Library/Frameworks/"
            "ApplicationServices.framework/ApplicationServices"
        )
        is_trusted = framework.AXIsProcessTrusted
        is_trusted.argtypes = []
        is_trusted.restype = ctypes.c_bool
        return bool(is_trusted())
    except Exception:  # noqa: BLE001 - unavailable on non-macOS test hosts
        return False


def request_accessibility_access() -> bool:
    """Register Moodito and request Accessibility access from macOS."""
    if accessibility_access_granted():
        return True
    try:
        import objc
        from Foundation import NSDictionary

        framework = ctypes.CDLL(
            "/System/Library/Frameworks/"
            "ApplicationServices.framework/ApplicationServices"
        )
        check = framework.AXIsProcessTrustedWithOptions
        check.argtypes = [ctypes.c_void_p]
        check.restype = ctypes.c_bool
        options = NSDictionary.dictionaryWithObject_forKey_(
            True,
            "AXTrustedCheckOptionPrompt",
        )
        return bool(check(ctypes.c_void_p(objc.pyobjc_id(options))))
    except Exception:  # noqa: BLE001 - permission requests remain best-effort
        return False


def _open_applications_list() -> None:
    launchers = (
        ("/System/Applications/Apps.app", "Apps"),
        ("/System/Applications/Launchpad.app", "Launchpad"),
        ("/Applications/Launchpad.app", "Launchpad"),
    )
    app_name = next((name for path, name in launchers if os.path.exists(path)), None)
    command = (
        ["open", "-a", app_name]
        if app_name is not None
        else ["open", "/Applications"]
    )
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_system_shortcut(key_code: int, *, control: bool = False) -> bool:
    modifiers = " using control down" if control else ""
    script = (
        'tell application "System Events" to key code '
        f"{key_code}{modifiers}"
    )
    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def perform_gesture_action(gesture: str) -> bool:
    """Perform the macOS action mapped to ``gesture``."""
    try:
        if gesture not in GESTURE_KEYS:
            return False
        if gesture == "double_blink":
            _open_applications_list()
            return True
        if gesture == "long_blink":
            if not accessibility_access_granted():
                return False
            return _run_system_shortcut(103)
        shortcuts = {
            "tilt_left": 123,
            "tilt_right": 124,
            "tilt_up": 126,
            "tilt_down": 125,
        }
        key_code = shortcuts.get(gesture)
        if key_code is None:
            return False
        if not accessibility_access_granted():
            return False
        return _run_system_shortcut(key_code, control=True)
    except Exception:  # noqa: BLE001 - action failures must not stop tracking
        return False


class GestureDetector:
    """Recognise configured gestures from a timestamped face-result stream."""

    def __init__(self, enabled: Mapping[str, bool] | None = None) -> None:
        self._enabled = dict(DEFAULT_GESTURES)
        self.set_enabled(enabled or DEFAULT_GESTURES)
        self._command_active = False
        self._baseline_pose: tuple[float, float] | None = None
        self._eyes_closed_since: float | None = None
        self._long_blink_seen = False
        self._blink_times: deque[float] = deque()
        self._pose_candidate: str | None = None
        self._pose_candidate_since: float | None = None
        self._pose_latched: str | None = None
        self._last_event_at = -math.inf

    def set_enabled(self, enabled: Mapping[str, bool]) -> None:
        self._enabled = {
            key: enabled.get(key, DEFAULT_GESTURES[key]) is True
            for key in GESTURE_KEYS
        }

    def reset(self, *, preserve_pose_baseline: bool = False) -> None:
        """Reset gesture state, optionally retaining pose calibration."""
        baseline_pose = self._baseline_pose if preserve_pose_baseline else None
        self._command_active = baseline_pose is not None
        self._baseline_pose = baseline_pose
        self._eyes_closed_since = None
        self._long_blink_seen = False
        self._blink_times.clear()
        self._pose_candidate = None
        self._pose_candidate_since = None
        self._pose_latched = None

    def update(
        self,
        timestamp: float,
        command_down: bool,
        blendshapes: Mapping[str, float],
        pose: tuple[float, float] | None,
    ) -> str | None:
        """Consume one face sample and return a newly recognised gesture."""
        if not command_down:
            self.reset()
            return None
        if not self._command_active:
            self._command_active = True
            self._baseline_pose = pose
        elif self._baseline_pose is None and pose is not None:
            self._baseline_pose = pose

        event = self._update_blinks(timestamp, blendshapes)
        if event is not None:
            return event
        if pose is None or self._baseline_pose is None:
            return None

        relative_pose = tuple(
            current - baseline
            for current, baseline in zip(pose, self._baseline_pose)
        )
        return self._update_held_pose(timestamp, relative_pose)

    def _emit(self, gesture: str, timestamp: float) -> str | None:
        if not self._enabled.get(gesture, False):
            return None
        if timestamp - self._last_event_at < _GESTURE_COOLDOWN_SECONDS:
            return None
        self._last_event_at = timestamp
        self._pose_candidate = None
        self._pose_candidate_since = None
        return gesture

    def _update_blinks(
        self,
        timestamp: float,
        blendshapes: Mapping[str, float],
    ) -> str | None:
        eyes_closed = min(
            blendshapes.get("eyeBlinkLeft", 0.0),
            blendshapes.get("eyeBlinkRight", 0.0),
        ) >= _BLINK_THRESHOLD
        if eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = timestamp
                self._long_blink_seen = False
            if (
                not self._long_blink_seen
                and timestamp - self._eyes_closed_since >= _LONG_BLINK_SECONDS
            ):
                self._long_blink_seen = True
                self._blink_times.clear()
                return self._emit("long_blink", timestamp)
            return None

        if self._eyes_closed_since is None:
            return None
        closed_for = timestamp - self._eyes_closed_since
        self._eyes_closed_since = None
        if self._long_blink_seen or closed_for < _BLINK_MIN_SECONDS:
            self._long_blink_seen = False
            return None

        self._blink_times.append(timestamp)
        while self._blink_times and timestamp - self._blink_times[0] > _DOUBLE_BLINK_WINDOW:
            self._blink_times.popleft()
        if len(self._blink_times) < 2:
            return None
        self._blink_times.clear()
        return self._emit("double_blink", timestamp)

    def _update_held_pose(
        self,
        timestamp: float,
        pose: tuple[float, float],
    ) -> str | None:
        pitch, roll = pose
        release_pitch = _PITCH_THRESHOLD_DEGREES * _POSE_RELEASE_RATIO
        release_roll = _ROLL_THRESHOLD_DEGREES * _POSE_RELEASE_RATIO
        if self._update_pose_latch(
            pitch,
            roll,
            release_pitch,
            release_roll,
        ):
            return None
        if abs(pitch) <= release_pitch and abs(roll) <= release_roll:
            self._pose_candidate = None
            self._pose_candidate_since = None
            return None

        candidate = None
        if roll <= -_ROLL_THRESHOLD_DEGREES:
            candidate = "tilt_left"
        elif roll >= _ROLL_THRESHOLD_DEGREES:
            candidate = "tilt_right"
        elif pitch >= _PITCH_THRESHOLD_DEGREES:
            candidate = "tilt_up"
        elif pitch <= -_PITCH_THRESHOLD_DEGREES:
            candidate = "tilt_down"
        if candidate is None:
            return None
        if candidate != self._pose_candidate:
            self._pose_candidate = candidate
            self._pose_candidate_since = timestamp
            return None
        if (
            self._pose_candidate_since is None
            or timestamp - self._pose_candidate_since
            < (
                _PITCH_HOLD_SECONDS
                if candidate in ("tilt_up", "tilt_down")
                else _ROLL_HOLD_SECONDS
            )
        ):
            return None
        self._pose_latched = candidate
        return self._emit(candidate, timestamp)

    def _update_pose_latch(
        self,
        pitch: float,
        roll: float,
        release_pitch: float,
        release_roll: float,
    ) -> bool:
        latched = self._pose_latched
        if latched is None:
            return False
        is_pitch = latched in ("tilt_up", "tilt_down")
        release_axis = pitch if is_pitch else roll
        release_threshold = release_pitch if is_pitch else release_roll
        if abs(release_axis) <= release_threshold:
            self._pose_candidate = None
            self._pose_candidate_since = None
            self._pose_latched = None
        return True