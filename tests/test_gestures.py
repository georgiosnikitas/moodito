"""Unit tests for face-gesture recognition and action mapping."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

import gestures


NEUTRAL_POSE = (0.0, 0.0)


def _rotation_x(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cosine, -sine, 0.0],
        [0.0, sine, cosine, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotation_z(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        [cosine, -sine, 0.0, 0.0],
        [sine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


class TestGestureSpecs:
    def test_contains_requested_rows_in_display_order(self) -> None:
        assert [spec.face_gesture for spec in gestures.GESTURE_SPECS] == [
            "😉 Double blink",
            "😌 Long blink",
            "↙️ Tilt head left",
            "↘️ Tilt head right",
            "⬆️ Tilt head up",
            "⬇️ Tilt head down",
        ]
        assert [spec.shortcut for spec in gestures.GESTURE_SPECS] == [
            "Apps",
            "F11 / Show Desktop",
            "Ctrl + ←",
            "Ctrl + →",
            "Ctrl + ↑",
            "Ctrl + ↓",
        ]


class TestPoseAngles:
    def test_identity_is_neutral(self) -> None:
        identity = _rotation_x(0.0)
        assert gestures.pose_angles_from_matrix(identity) == pytest.approx(
            NEUTRAL_POSE
        )

    def test_extracts_nose_up_pitch_and_roll(self) -> None:
        assert gestures.pose_angles_from_matrix(_rotation_x(-18.0)) == pytest.approx(
            (18.0, 0.0)
        )
        assert gestures.pose_angles_from_matrix(_rotation_x(18.0)) == pytest.approx(
            (-18.0, 0.0)
        )
        assert gestures.pose_angles_from_matrix(_rotation_z(-16.0)) == pytest.approx(
            (0.0, -16.0)
        )

    def test_invalid_matrix_returns_none(self) -> None:
        assert gestures.pose_angles_from_matrix([]) is None
        assert gestures.pose_angles_from_matrix([["bad"] * 3] * 3) is None


class TestGestureDetector:
    @pytest.mark.parametrize(
        ("matrix_degrees", "expected"),
        [(-18.0, "tilt_up"), (18.0, "tilt_down")],
    )
    def test_physical_pitch_matrix_detects_direction(
        self, matrix_degrees, expected
    ) -> None:
        detector = gestures.GestureDetector()
        neutral = gestures.pose_angles_from_matrix(_rotation_x(0.0))
        tilted = gestures.pose_angles_from_matrix(_rotation_x(matrix_degrees))

        assert detector.update(0.0, True, {}, neutral) is None
        assert detector.update(0.1, True, {}, tilted) is None
        assert detector.update(1.1, True, {}, tilted) == expected

    def test_requires_command_and_calibrates_from_press_pose(self) -> None:
        detector = gestures.GestureDetector()
        assert detector.update(0.0, False, {}, NEUTRAL_POSE) is None
        assert detector.update(0.1, True, {}, (8.0, -4.0)) is None
        assert detector.update(0.2, True, {}, (8.0, 10.0)) is None
        assert detector.update(0.8, True, {}, (8.0, 10.0)) == "tilt_right"

    def test_calibrates_when_pose_arrives_after_command_press(self) -> None:
        detector = gestures.GestureDetector()
        assert detector.update(0.0, True, {}, None) is None
        assert detector.update(0.1, True, {}, NEUTRAL_POSE) is None
        assert detector.update(0.2, True, {}, (0.0, 14.0)) is None
        assert detector.update(0.8, True, {}, (0.0, 14.0)) == "tilt_right"

    def test_detects_double_blink(self) -> None:
        detector = gestures.GestureDetector()
        closed = {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.9}
        detector.update(0.0, True, {}, NEUTRAL_POSE)
        detector.update(0.10, True, closed, NEUTRAL_POSE)
        assert detector.update(0.20, True, {}, NEUTRAL_POSE) is None
        detector.update(0.40, True, closed, NEUTRAL_POSE)
        assert detector.update(0.50, True, {}, NEUTRAL_POSE) == "double_blink"

    def test_long_blink_wins_and_does_not_seed_double_blink(self) -> None:
        detector = gestures.GestureDetector()
        closed = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}
        detector.update(0.0, True, {}, NEUTRAL_POSE)
        detector.update(0.10, True, closed, NEUTRAL_POSE)
        assert detector.update(0.95, True, closed, NEUTRAL_POSE) == "long_blink"
        assert detector.update(1.00, True, {}, NEUTRAL_POSE) is None
        detector.update(1.80, True, closed, NEUTRAL_POSE)
        assert detector.update(1.90, True, {}, NEUTRAL_POSE) is None

    @pytest.mark.parametrize(
        "pose,held_until,expected",
        [
            ((0.0, -14.0), 0.7, "tilt_left"),
            ((0.0, 14.0), 0.7, "tilt_right"),
            ((13.0, 0.0), 1.2, "tilt_up"),
            ((-13.0, 0.0), 1.2, "tilt_down"),
        ],
    )
    def test_detects_held_head_pose(self, pose, held_until, expected) -> None:
        detector = gestures.GestureDetector()
        detector.update(0.0, True, {}, NEUTRAL_POSE)
        assert detector.update(0.1, True, {}, pose) is None
        assert detector.update(held_until, True, {}, pose) == expected
        assert detector.update(held_until + 0.3, True, {}, pose) is None

    def test_pitch_gesture_releases_despite_small_roll_offset(self) -> None:
        detector = gestures.GestureDetector()
        detector.update(0.0, True, {}, NEUTRAL_POSE)
        detector.update(0.1, True, {}, (13.0, 0.0))
        assert detector.update(1.1, True, {}, (13.0, 0.0)) == "tilt_up"

        assert detector.update(1.3, True, {}, (0.0, 7.0)) is None
        assert detector.update(1.5, True, {}, (-13.0, 0.0)) is None
        assert detector.update(2.5, True, {}, (-13.0, 0.0)) == "tilt_down"

    def test_disabled_gesture_is_not_emitted(self) -> None:
        detector = gestures.GestureDetector({"tilt_right": False})
        detector.update(0.0, True, {}, NEUTRAL_POSE)
        detector.update(0.1, True, {}, (0.0, 14.0))
        assert detector.update(0.8, True, {}, (0.0, 14.0)) is None

    def test_command_release_resets_partial_gesture(self) -> None:
        detector = gestures.GestureDetector()
        closed = {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}
        detector.update(0.0, True, {}, NEUTRAL_POSE)
        detector.update(0.1, True, closed, NEUTRAL_POSE)
        detector.update(0.2, True, {}, NEUTRAL_POSE)
        detector.update(0.3, False, {}, NEUTRAL_POSE)
        detector.update(0.4, True, {}, NEUTRAL_POSE)
        detector.update(0.5, True, closed, NEUTRAL_POSE)
        assert detector.update(0.6, True, {}, NEUTRAL_POSE) is None


class TestMacActions:
    @pytest.mark.parametrize(
        "gesture,key_code",
        [
            ("tilt_left", 123),
            ("tilt_right", 124),
            ("tilt_up", 126),
            ("tilt_down", 125),
        ],
    )
    def test_runs_control_system_shortcuts(
        self, gesture, key_code, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            gestures, "accessibility_access_granted", lambda: True
        )
        monkeypatch.setattr(
            gestures,
            "_run_system_shortcut",
            lambda code, *, control=False: calls.append((code, control)) or True,
        )
        assert gestures.perform_gesture_action(gesture) is True
        assert calls == [(key_code, True)]

    def test_long_blink_runs_show_desktop_shortcut(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            gestures, "accessibility_access_granted", lambda: True
        )
        monkeypatch.setattr(
            gestures,
            "_run_system_shortcut",
            lambda code, *, control=False: calls.append((code, control)) or True,
        )
        assert gestures.perform_gesture_action("long_blink") is True
        assert calls == [(103, False)]

    @pytest.mark.parametrize(
        "control,expected_script",
        [
            (False, 'tell application "System Events" to key code 103'),
            (
                True,
                'tell application "System Events" to key code 126 using control down',
            ),
        ],
    )
    def test_system_shortcut_runs_osascript(
        self, control, expected_script, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            gestures.subprocess,
            "Popen",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        assert gestures._run_system_shortcut(126 if control else 103, control=control)
        assert calls[0][0][0] == ["osascript", "-e", expected_script]
        assert calls[0][1] == {
            "stdout": gestures.subprocess.DEVNULL,
            "stderr": gestures.subprocess.DEVNULL,
        }

    @pytest.mark.parametrize(
        "available_path,app_name",
        [
            ("/System/Applications/Apps.app", "Apps"),
            ("/System/Applications/Launchpad.app", "Launchpad"),
        ],
    )
    def test_double_blink_opens_available_app_list(
        self, available_path, app_name, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            gestures.os.path,
            "exists",
            lambda path: path == available_path,
        )
        monkeypatch.setattr(
            gestures.subprocess,
            "Popen",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        assert gestures.perform_gesture_action("double_blink") is True
        assert calls[0][0][0] == ["open", "-a", app_name]

    def test_double_blink_falls_back_to_applications_folder(
        self, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(gestures.os.path, "exists", lambda _path: False)
        monkeypatch.setattr(
            gestures.subprocess,
            "Popen",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        assert gestures.perform_gesture_action("double_blink") is True
        assert calls[0][0][0] == ["open", "/Applications"]

    def test_unknown_gesture_is_rejected(self) -> None:
        assert gestures.perform_gesture_action("unknown") is False

    def test_command_key_reads_modifier_flags_without_input_monitoring(
        self, monkeypatch
    ) -> None:
        fake_appkit = SimpleNamespace(
            NSEvent=SimpleNamespace(modifierFlags=lambda: 1 << 20),
            NSEventModifierFlagCommand=1 << 20,
        )
        monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)

        assert gestures.command_key_down() is True

    def test_system_shortcut_is_blocked_without_accessibility(
        self, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            gestures, "accessibility_access_granted", lambda: False
        )
        monkeypatch.setattr(
            gestures,
            "_run_system_shortcut",
            lambda *args, **kwargs: calls.append((args, kwargs)) or True,
        )
        assert gestures.perform_gesture_action("tilt_up") is False
        assert calls == []

    def test_accessibility_status_uses_native_ax_api(self, monkeypatch) -> None:
        class FakeTrustedFunction:
            argtypes = None
            restype = None

            def __call__(self):
                return True

        trusted = FakeTrustedFunction()
        framework = SimpleNamespace(AXIsProcessTrusted=trusted)
        monkeypatch.setattr(gestures.ctypes, "CDLL", lambda _path: framework)

        assert gestures.accessibility_access_granted() is True
        assert trusted.argtypes == []
        assert trusted.restype is gestures.ctypes.c_bool

    def test_accessibility_request_uses_prompt_option(self, monkeypatch) -> None:
        calls = []

        class FakeCheck:
            argtypes = None
            restype = None

            def __call__(self, pointer):
                calls.append(pointer.value)
                return True

        check = FakeCheck()
        framework = SimpleNamespace(AXIsProcessTrustedWithOptions=check)
        options = object()
        foundation = type(sys)("Foundation")
        foundation.NSDictionary = SimpleNamespace(
            dictionaryWithObject_forKey_=(
                lambda value, key: calls.append((value, key)) or options
            )
        )
        objc = type(sys)("objc")
        objc.pyobjc_id = lambda value: 123 if value is options else 0
        monkeypatch.setitem(sys.modules, "Foundation", foundation)
        monkeypatch.setitem(sys.modules, "objc", objc)
        monkeypatch.setattr(gestures.ctypes, "CDLL", lambda _path: framework)
        monkeypatch.setattr(
            gestures, "accessibility_access_granted", lambda: False
        )

        assert gestures.request_accessibility_access() is True
        assert calls == [(True, "AXTrustedCheckOptionPrompt"), 123]
        assert check.argtypes == [gestures.ctypes.c_void_p]
        assert check.restype is gestures.ctypes.c_bool
