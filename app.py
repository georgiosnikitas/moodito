"""Moodito — a macOS menu bar app that recognises your emotions.

It captures frames from the webcam, runs Google's MediaPipe Face
Landmarker to extract facial blendshapes, maps them to a coarse emotion,
and shows it as an emoji + label in the menu bar title.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import urllib.request

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
# Store the model in a writable per-user directory so it works both when run
# from source and when packaged as a read-only .app bundle.
DATA_DIR = os.path.expanduser("~/Library/Application Support/Moodito")
MODEL_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)

# How often (seconds) the menu bar title is refreshed from the latest result.
UI_REFRESH_INTERVAL = 0.3
# Target webcam sampling rate (seconds between processed frames).
SAMPLE_INTERVAL = 0.15
# Spinner frames cycled while the app is still starting up.
LOADING_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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
        super().__init__("Moodito", title="⠋ loading…", quit_button=None)
        self._worker = FaceWorker()
        self._paused = False
        # Display mode: False = emoji + label text, True = Moodito icon only.
        self._icon_only = False
        self._showing_icon = False
        self._icon_path = resource_path(MENUBAR_ICON)
        # Animation frame counter for the startup spinner.
        self._loading_i = 0

        self.menu = [
            rumps.MenuItem("Detected: …", callback=None),
            None,
            rumps.MenuItem("Show icon only", callback=self.toggle_icon_only),
            rumps.MenuItem("Camera Grant Access", callback=self.grant_camera),
            rumps.MenuItem("Pause", callback=self.toggle_pause),
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        self._detected_item = self.menu["Detected: …"]
        self._pause_item = self.menu["Pause"]
        self._icon_only_item = self.menu["Show icon only"]
        self._camera_item = self.menu["Camera Grant Access"]

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
            return

        error = self._worker.error
        if error:
            self._render_status("⚠️ Moodito", allow_icon=False)
            self._detected_item.title = f"Error: {error}"
            return

        if not self._worker.ready:
            # Still starting up (model load + camera open): show a spinner.
            self._loading_i += 1
            frame = LOADING_FRAMES[self._loading_i % len(LOADING_FRAMES)]
            self._render_status(f"{frame} loading…", allow_icon=False)
            self._detected_item.title = "Loading…"
            return

        result = self._worker.result
        self._render_status(result.title, allow_icon=True)
        self._detected_item.title = f"Detected: {result.label} ({result.score:.0%})"

    def toggle_icon_only(self, sender) -> None:
        self._icon_only = not self._icon_only
        sender.state = self._icon_only
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

    def quit_app(self, _sender) -> None:
        self._worker.stop()
        rumps.quit_application()


def main() -> None:
    # Validate the emoji table is wired up (cheap sanity check at startup).
    assert "happy" in EMOTION_EMOJI
    # Ask for camera access up front so macOS shows the permission prompt.
    request_camera_access()
    MooditoApp().run()


if __name__ == "__main__":
    main()
