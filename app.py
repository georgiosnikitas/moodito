"""Moodito — a macOS menu bar app that recognises your emotions.

It captures frames from the webcam, runs Google's MediaPipe Face
Landmarker to extract facial blendshapes, maps them to a coarse emotion,
and shows it as an emoji + label in the menu bar title.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime

import cv2
import mediapipe as mp
import rumps
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from emotion import EMOTION_EMOJI, EmotionResult, infer_emotion

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# Colored app icon shown in the menu bar when "icon only" mode is enabled.
MENUBAR_ICON = "moodito.png"
# Support / tip jar link opened from the menu.
BMC_URL = "https://buymeacoffee.com/georgiosnikitas"
# QR code image (bundled) shown under the Buy Me a Coffee menu item.
BMC_QR = "bmc_qr.png"
# Store the model in a writable per-user directory so it works both when run
# from source and when packaged as a read-only .app bundle.
DATA_DIR = os.path.expanduser("~/Library/Application Support/Moodito")
MODEL_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)
# Persisted user preferences (e.g. the "icon only" display mode).
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
# Persisted usage statistics (time + occurrences per emotion).
STATS_PATH = os.path.join(DATA_DIR, "stats.json")
# Emotions accumulated in the statistics.
TRACKED_EMOTIONS = ["happy", "sad", "surprised", "angry", "neutral", "no face"]
# Non-emotion states that are also tracked.
EXTRA_STATES = ["paused", "error"]
# All statistic rows, in display order.
STAT_KEYS = TRACKED_EMOTIONS + EXTRA_STATES
# Emoji shown for each statistic row (emotions reuse EMOTION_EMOJI).
STAT_EMOJI = {**EMOTION_EMOJI, "paused": "⏸️", "error": "⚠️"}

# How often (seconds) the menu bar title is refreshed from the latest result.
UI_REFRESH_INTERVAL = 0.3
# Target webcam sampling rate (seconds between processed frames).
SAMPLE_INTERVAL = 0.15


def resource_path(name: str) -> str:
    """Resolve a bundled resource path for both source and PyInstaller runs."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def ensure_model() -> None:
    """Download the MediaPipe Face Landmarker model if it is not present."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_settings() -> dict:
    """Load persisted preferences, returning an empty dict if none exist."""
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    """Persist preferences to disk (best-effort)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(settings, fh)
    except OSError:
        pass


def load_stats() -> tuple[dict, str | None]:
    """Load persisted statistics and the tracking start timestamp.

    Returns (per-emotion stats normalised for all emotions, started_at iso str).
    """
    raw: dict = {}
    try:
        with open(STATS_PATH, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            raw = loaded
    except (OSError, ValueError):
        pass
    # Support both the wrapped format and the older flat emotion-only format.
    emotions_raw = raw.get("emotions") if isinstance(raw.get("emotions"), dict) else raw
    started_at = raw.get("started_at") if isinstance(raw.get("started_at"), str) else None
    stats = {}
    for emotion in STAT_KEYS:
        entry = emotions_raw.get(emotion) if isinstance(emotions_raw.get(emotion), dict) else {}
        stats[emotion] = {
            "seconds": float(entry.get("seconds", 0.0)),
            "count": int(entry.get("count", 0)),
        }
    return stats, started_at


def save_stats(stats: dict, started_at: str | None) -> None:
    """Persist statistics and start timestamp to disk (best-effort)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(STATS_PATH, "w", encoding="utf-8") as fh:
            json.dump({"started_at": started_at, "emotions": stats}, fh)
    except OSError:
        pass


def format_timestamp(iso: str | None) -> str:
    """Format an ISO timestamp as 'Jun 21, 2026 · 22:10' (best-effort)."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %Y · %H:%M")
    except ValueError:
        return iso


def format_bytes(num: int) -> str:
    """Format a byte count as a compact human string (e.g. '1.2 KB')."""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def stats_file_size() -> int:
    """Return the size in bytes of the persisted statistics file (0 if absent)."""
    try:
        return os.path.getsize(STATS_PATH)
    except OSError:
        return 0


def format_duration(seconds: float) -> str:
    """Format a duration as a compact human string (e.g. '1h 03m', '2m 05s')."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def set_monospaced_title(item, text: str) -> None:
    """Set a menu item's title in a monospaced font so columns stay aligned.

    Falls back to a plain title if AppKit is unavailable (non-bundled run).
    """
    try:
        from AppKit import NSFont, NSFontAttributeName
        from Foundation import NSAttributedString

        font = NSFont.monospacedSystemFontOfSize_weight_(
            NSFont.systemFontSize(), 0.0
        )
        attributed = NSAttributedString.alloc().initWithString_attributes_(
            text, {NSFontAttributeName: font}
        )
        item._menuitem.setAttributedTitle_(attributed)
    except Exception:  # noqa: BLE001 - optional AppKit dependency
        item.title = text


def request_camera_access() -> None:
    """Trigger the macOS camera permission prompt on the main thread.

    OpenCV opens the camera on a background thread, where macOS will not show
    the TCC authorization dialog. Asking AVFoundation explicitly (from the main
    thread, at startup) makes the prompt appear so the user can grant access.
    Best-effort: if AVFoundation is unavailable, the worker still retries.
    """
    if camera_authorization_status() != 0:
        # Only the "not determined" state (0) can show a prompt.
        return
    try:
        import AVFoundation  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional dependency / non-bundled run
        return
    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVFoundation.AVMediaTypeVideo, lambda _granted: None
    )


def camera_authorization_status() -> int | None:
    """Return the AVFoundation camera authorization status, or None if unknown.

    0 = not determined, 1 = restricted, 2 = denied, 3 = authorized.
    """
    try:
        import AVFoundation  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional dependency / non-bundled run
        return None
    return AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        AVFoundation.AVMediaTypeVideo
    )


def open_camera_settings() -> None:
    """Open System Settings at the Privacy > Camera pane."""
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"],
        check=False,
    )


class FaceWorker(threading.Thread):
    """Background thread: capture frames and infer the current emotion."""

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__(daemon=True)
        self._camera_index = camera_index
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._result = EmotionResult("neutral", 0.0)
        self._error: str | None = None
        self._ready = False

    @property
    def result(self) -> EmotionResult:
        with self._lock:
            return self._result

    @property
    def ready(self) -> bool:
        """True once the first frame has been processed (startup complete)."""
        with self._lock:
            return self._ready

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def _set_result(self, result: EmotionResult) -> None:
        with self._lock:
            self._result = result
            self._error = None
            self._ready = True

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            ensure_model()
        except Exception as exc:  # noqa: BLE001 - surface any download failure
            self._set_error(f"model download failed: {exc}")
            return

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            output_face_blendshapes=True,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )

        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while not self._stop.is_set():
                cap = cv2.VideoCapture(self._camera_index)
                if not cap.isOpened():
                    cap.release()
                    self._set_error("please grant access to camera")
                    self._stop.wait(2.0)
                    continue

                self._run_capture_loop(cap, landmarker)
                cap.release()

    def _run_capture_loop(self, cap, landmarker) -> None:
        timestamp_ms = 0
        failures = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                failures += 1
                # A few read failures can be transient; many means the camera
                # was lost, so break out to reopen it.
                if failures > 10:
                    self._set_error("lost camera, reconnecting…")
                    return
                self._stop.wait(0.2)
                continue
            failures = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += int(SAMPLE_INTERVAL * 1000)
            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection.face_blendshapes:
                scores = {
                    category.category_name: category.score
                    for category in detection.face_blendshapes[0]
                }
                self._set_result(infer_emotion(scores))
            else:
                self._set_result(EmotionResult("no face", 0.0))

            self._stop.wait(SAMPLE_INTERVAL)


class MooditoApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Moodito", title="😐 neutral", quit_button=None)
        self._worker = FaceWorker()
        self._paused = False
        # Display mode: False = emoji + label text, True = Moodito icon only.
        # Restored from persisted settings so the choice survives restarts.
        self._settings = load_settings()
        self._icon_only = bool(self._settings.get("icon_only", False))
        self._showing_icon = False
        self._icon_path = resource_path(MENUBAR_ICON)
        # Persisted per-emotion usage statistics.
        self._stats, self._stats_started_at = load_stats()
        if self._stats_started_at is None:
            # First run: record when tracking began.
            self._stats_started_at = datetime.now().isoformat(timespec="seconds")
            save_stats(self._stats, self._stats_started_at)
        self._last_emotion: str | None = None
        self._ticks_since_save = 0

        # Build the live Statistics submenu (one row per tracked state).
        self._stats_menu = rumps.MenuItem("Statistics")
        self._stats_since_item = rumps.MenuItem("Since …", callback=None)
        self._stats_menu.add(self._stats_since_item)
        self._stats_menu.add(None)
        self._stats_header_item = rumps.MenuItem("Header", callback=None)
        self._stats_menu.add(self._stats_header_item)
        self._stats_items: dict[str, rumps.MenuItem] = {}
        for key in STAT_KEYS:
            item = rumps.MenuItem(key, callback=None)
            self._stats_menu.add(item)
            self._stats_items[key] = item
        self._stats_menu.add(None)
        self._stats_total_item = rumps.MenuItem("Total", callback=None)
        self._stats_menu.add(self._stats_total_item)
        self._stats_menu.add(None)
        self._stats_reset_item = rumps.MenuItem(
            "Reset Statistics", callback=self.reset_stats
        )
        self._stats_menu.add(self._stats_reset_item)

        self.menu = [
            rumps.MenuItem("Detected: …", callback=None),
            None,
            self._stats_menu,
            None,
            rumps.MenuItem("Show icon only", callback=self.toggle_icon_only),
            rumps.MenuItem("Camera Grant Access", callback=self.grant_camera),
            rumps.MenuItem("Pause", callback=self.toggle_pause),
            None,
            rumps.MenuItem("Buy Me a Coffee ☕", callback=self.buy_me_a_coffee),
            rumps.MenuItem(
                "",
                icon=resource_path(BMC_QR),
                dimensions=[180, 180],
                callback=self.buy_me_a_coffee,
            ),
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        self._detected_item = self.menu["Detected: …"]
        self._pause_item = self.menu["Pause"]
        self._icon_only_item = self.menu["Show icon only"]
        self._camera_item = self.menu["Camera Grant Access"]
        # Reflect the restored display mode in the menu item's checkmark.
        self._icon_only_item.state = self._icon_only
        self._update_stats_menu()

        self._worker.start()

    def _render_status(self, title_text: str, allow_icon: bool) -> None:
        """Update the menu bar to show either the icon or the title text.

        `allow_icon` is False for transient states (error, paused) so they are
        always shown as text regardless of the chosen display mode.
        """
        if self._icon_only and allow_icon:
            if not self._showing_icon:
                # Set the icon BEFORE clearing the title, otherwise rumps
                # momentarily has neither and falls back to the app name
                # ("Moodito"), which then sticks alongside the icon.
                self.icon = self._icon_path
                self.title = None
                self._showing_icon = True
        else:
            # Set the title BEFORE removing the icon for the same reason.
            self.title = title_text
            if self._showing_icon:
                self.icon = None
                self._showing_icon = False

    @rumps.timer(UI_REFRESH_INTERVAL)
    def refresh(self, _timer) -> None:
        # Hide the grant-access item once camera permission is authorized.
        self._camera_item.hidden = camera_authorization_status() == 3

        if self._paused:
            self._accumulate_stats("paused")
            return

        error = self._worker.error
        if error:
            self._render_status("⚠️ Moodito", allow_icon=False)
            self._detected_item.title = f"Error: {error}"
            self._accumulate_stats("error")
            return

        result = self._worker.result
        self._render_status(result.title, allow_icon=True)
        self._detected_item.title = f"Detected: {result.label} ({result.score:.0%})"
        self._accumulate_stats(result.label)

    def _accumulate_stats(self, label: str) -> None:
        """Add elapsed time to the current emotion and persist periodically."""
        if label in self._stats:
            self._stats[label]["seconds"] += UI_REFRESH_INTERVAL
            # Count a new occurrence each time the emotion changes.
            if label != self._last_emotion:
                self._stats[label]["count"] += 1
            self._update_stats_menu()
        self._last_emotion = label

        # Flush to disk roughly every 10 seconds to limit write frequency.
        self._ticks_since_save += 1
        if self._ticks_since_save * UI_REFRESH_INTERVAL >= 10:
            self._ticks_since_save = 0
            save_stats(self._stats, self._stats_started_at)

    def _update_stats_menu(self) -> None:
        """Refresh the Statistics submenu rows as an aligned table."""
        # Column header (leading spaces account for the emoji column width).
        header = (
            f"{'':<4}{'Emotion':<9}"
            f"{'%':>5}"
            f"{'Time':>9}"
            f"{'Count':>7}"
        )
        set_monospaced_title(self._stats_header_item, header)
        total = sum(entry["seconds"] for entry in self._stats.values())
        total_count = sum(entry["count"] for entry in self._stats.values())
        for key, item in self._stats_items.items():
            entry = self._stats[key]
            pct = (entry["seconds"] / total * 100.0) if total else 0.0
            emoji = STAT_EMOJI.get(key, "")
            # Fixed-width columns: name | percent | duration | count.
            row = (
                f"{emoji}  {key:<9}"
                f"{pct:>4.0f}%"
                f"{format_duration(entry['seconds']):>9}"
                f"{'×' + str(entry['count']):>7}"
            )
            set_monospaced_title(item, row)
        # Totals row, aligned with the same columns.
        total_row = (
            f"Σ  {'Total':<9}"
            f"{100 if total else 0:>4.0f}%"
            f"{format_duration(total):>9}"
            f"{'×' + str(total_count):>7}"
        )
        set_monospaced_title(self._stats_total_item, total_row)
        self._stats_since_item.title = f"Since {format_timestamp(self._stats_started_at)}"
        self._stats_reset_item.title = (
            f"Reset Statistics ({format_bytes(stats_file_size())})"
        )

    def toggle_icon_only(self, sender) -> None:
        self._icon_only = not self._icon_only
        sender.state = self._icon_only
        # Persist the choice so it is restored on the next launch.
        self._settings["icon_only"] = self._icon_only
        save_settings(self._settings)
        # Apply immediately rather than waiting for the next refresh tick.
        if not self._paused:
            self.refresh(None)

    def toggle_pause(self, _sender) -> None:
        self._paused = not self._paused
        if self._paused:
            self._pause_item.title = "Resume"
            self._render_status("⏸️ Moodito", allow_icon=False)
            self._detected_item.title = "Paused"
        else:
            self._pause_item.title = "Pause"

    def grant_camera(self, _sender) -> None:
        status = camera_authorization_status()
        if status is None:
            rumps.alert("Camera", "Camera control is unavailable in this build.")
        elif status == 0:
            # Not determined yet → trigger the macOS permission prompt.
            request_camera_access()
        elif status == 3:
            rumps.alert("Camera", "Camera access is already granted. ✅")
        else:
            # Denied or restricted → can't re-prompt, open Settings instead.
            open_camera_settings()

    def buy_me_a_coffee(self, _sender) -> None:
        """Open the Buy Me a Coffee page in the default browser."""
        subprocess.run(["open", BMC_URL], check=False)

    def reset_stats(self, _sender) -> None:
        """Clear all accumulated statistics."""
        self._stats = {
            key: {"seconds": 0.0, "count": 0} for key in STAT_KEYS
        }
        self._last_emotion = None
        self._stats_started_at = datetime.now().isoformat(timespec="seconds")
        save_stats(self._stats, self._stats_started_at)
        self._update_stats_menu()

    def quit_app(self, _sender) -> None:
        self._worker.stop()
        save_stats(self._stats, self._stats_started_at)
        rumps.quit_application()


def main() -> None:
    # Validate the emoji table is wired up (cheap sanity check at startup).
    assert "happy" in EMOTION_EMOJI
    # Ask for camera access up front so macOS shows the permission prompt.
    request_camera_access()
    MooditoApp().run()


if __name__ == "__main__":
    main()
