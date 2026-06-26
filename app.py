"""Moodito — a macOS menu bar app that recognises your emotions.

It captures frames from the webcam, runs Google's MediaPipe Face
Landmarker to extract facial blendshapes, maps them to a coarse emotion,
and shows it as an emoji + label in the menu bar title.
"""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import rumps
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from emotion import (
    DEFAULT_SENSITIVITY,
    EMOTION_EMOJI,
    SENSITIVITY_EMOTIONS,
    SENSITIVITY_LEVELS,
    EmotionResult,
    infer_emotion,
)

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# Colored app icon shown in the menu bar when "icon only" mode is enabled.
MENUBAR_ICON = "assets/moodito.png"
# Support / tip jar link opened from the menu.
BMC_URL = "https://buymeacoffee.com/georgiosnikitas"
# QR code image (bundled) shown under the Buy Me a Coffee menu item.
BMC_QR = "assets/bmc_qr.png"
# Lemon Squeezy storefront where a license can be purchased.
LICENSE_BUY_URL = "https://georgiosnikitas.lemonsqueezy.com/"
# Lemon Squeezy customer portal used to look up past orders / license keys.
LICENSE_RESTORE_URL = "https://app.lemonsqueezy.com/my-orders"
# Lemon Squeezy License API base (separate from the main API; no auth needed).
LICENSE_API_BASE = "https://api.lemonsqueezy.com/v1/licenses"
# Network timeout (seconds) for license API calls.
LICENSE_API_TIMEOUT = 10
# Title shown on every license-related alert dialog.
LICENSE_ALERT_TITLE = "Moodito License"
# How often (seconds) an active license is re-validated in the background.
LICENSE_RECHECK_INTERVAL = 6 * 60 * 60
# Outcomes returned by validate_license().
LICENSE_VALID = "valid"  # confirmed active by Lemon Squeezy
LICENSE_INVALID = "invalid"  # reachable, but expired/disabled/deactivated
LICENSE_UNREACHABLE = "unreachable"  # network/server error (treated as transient)
# Store the model in a writable per-user directory so it works both when run
# from source and when packaged as a read-only .app bundle.
DATA_DIR = os.path.expanduser("~/Library/Application Support/Moodito")
MODEL_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)
# Persisted user preferences (e.g. the "icon only" display mode).
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
# Persisted license activation (key + Lemon Squeezy instance id).
LICENSE_PATH = os.path.join(DATA_DIR, "license.json")
# Persisted usage statistics (time + occurrences per emotion).
STATS_PATH = os.path.join(DATA_DIR, "stats.json")
# Raw per-sample detection log (one row per refresh tick).
RAW_PATH = os.path.join(DATA_DIR, "raw_data.csv")
# Column labels for the raw detection log.
RAW_HEADER = ["timestamp", "state", "score"]
# Emotions accumulated in the statistics.
TRACKED_EMOTIONS = ["happy", "sad", "surprised", "angry", "neutral", "no face"]
# Non-emotion states that are also tracked.
EXTRA_STATES = ["paused", "error"]
# All statistic rows, in display order.
STAT_KEYS = TRACKED_EMOTIONS + EXTRA_STATES
# Emoji shown for each statistic row (emotions reuse EMOTION_EMOJI).
STAT_EMOJI = {**EMOTION_EMOJI, "paused": "⏸️", "error": "⚠️"}
# Title shown on the statistics datetime-range prompt and its error alerts.
STATS_RANGE_TITLE = "Datetime Range"

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


def load_license() -> dict:
    """Load the stored license activation, or an empty dict if none exists."""
    try:
        with open(LICENSE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_license(license_data: dict) -> None:
    """Persist the license activation to disk (best-effort)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LICENSE_PATH, "w", encoding="utf-8") as fh:
            json.dump(license_data, fh)
    except OSError:
        pass


def clear_license() -> None:
    """Remove the stored license activation (best-effort)."""
    try:
        os.remove(LICENSE_PATH)
    except OSError:
        pass


def is_license_active(license_data: dict) -> bool:
    """True if the stored license has both a key and an activation instance."""
    return bool(license_data.get("license_key") and license_data.get("instance_id"))


def mask_license_key(key: str) -> str:
    """Mask a license key for display, revealing only the last 4 characters."""
    key = (key or "").strip()
    if not key:
        return "—"
    tail = key[-4:]
    return f"•••• {tail}"


def license_instance_name() -> str:
    """A human label sent to Lemon Squeezy to identify this activation."""
    try:
        host = socket.gethostname() or "Mac"
    except OSError:
        host = "Mac"
    return f"Moodito on {host}"


def _license_api_request(action: str, params: dict) -> dict:
    """POST to the Lemon Squeezy License API and return the parsed JSON body.

    Returns the decoded response for both success (2xx) and documented client
    errors (4xx), which also carry a JSON body with an ``error`` field.
    """
    url = f"{LICENSE_API_BASE}/{action}"
    body = urllib.parse.urlencode(params).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=LICENSE_API_TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            return json.load(exc)
        except ValueError:
            return {"error": f"HTTP {exc.code}"}


def activate_license(key: str, instance_name: str) -> tuple[bool, str, str]:
    """Activate a license key with Lemon Squeezy.

    Returns (ok, message, instance_id). On success ``instance_id`` is the
    non-empty Lemon Squeezy activation instance id.
    """
    try:
        data = _license_api_request(
            "activate", {"license_key": key, "instance_name": instance_name}
        )
    except (OSError, ValueError) as exc:
        return False, f"could not reach license server: {exc}", ""
    if not data.get("activated"):
        return False, str(data.get("error") or "activation failed"), ""
    instance = data.get("instance") if isinstance(data.get("instance"), dict) else {}
    instance_id = str(instance.get("id") or "")
    if not instance_id:
        return False, "activation succeeded but no instance id was returned", ""
    return True, "activated", instance_id


def validate_license(key: str, instance_id: str) -> str:
    """Validate a previously activated license key instance.

    Returns one of ``LICENSE_VALID`` (still active), ``LICENSE_INVALID``
    (reachable but expired/disabled/deactivated) or ``LICENSE_UNREACHABLE``
    (network/server error — caller should treat as transient).
    """
    try:
        data = _license_api_request(
            "validate", {"license_key": key, "instance_id": instance_id}
        )
    except (OSError, ValueError):
        return LICENSE_UNREACHABLE
    return LICENSE_VALID if data.get("valid") else LICENSE_INVALID


def deactivate_license(key: str, instance_id: str) -> tuple[bool, str]:
    """Deactivate a license key instance with Lemon Squeezy.

    Returns (ok, message).
    """
    try:
        data = _license_api_request(
            "deactivate", {"license_key": key, "instance_id": instance_id}
        )
    except (OSError, ValueError) as exc:
        return False, f"could not reach license server: {exc}"
    if data.get("deactivated"):
        return True, "deactivated"
    return False, str(data.get("error") or "deactivation failed")


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


def format_datetime(value: datetime) -> str:
    """Format a datetime as 'Jun 21, 2026 22:10'."""
    return value.strftime("%b %d, %Y %H:%M")


def format_timestamp(iso: str | None) -> str:
    """Format an ISO timestamp as 'Jun 21, 2026 22:10' (best-effort)."""
    if not iso:
        return "—"
    try:
        return format_datetime(datetime.fromisoformat(iso))
    except ValueError:
        return iso


def parse_iso_datetime(text: str | None) -> datetime | None:
    """Parse an ISO timestamp into a datetime, or None if it is invalid."""
    try:
        return datetime.fromisoformat(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_datetime_input(text: str | None) -> datetime | None:
    """Parse a user-entered date/time into a datetime, or None if invalid.

    Accepts 'YYYY-MM-DD HH:MM[:SS]' (space or 'T' separator) and 'YYYY-MM-DD'.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    candidate = cleaned.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return parse_iso_datetime(cleaned)


def parse_datetime_range(
    text: str | None,
) -> tuple[datetime | None, datetime | None] | None:
    """Parse a 'START to END' range into (start, end), or None if invalid.

    The two ends may be separated by ' to ', '→', '–', '—' or '..'. A start of
    'begin' yields ``None``, meaning the "Since" (data start) datetime. An end
    of 'now' (or an empty end) yields ``None``, meaning a live window.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    for sep in (" to ", "→", "–", "—", ".."):
        if sep in cleaned:
            left, right = cleaned.split(sep, 1)
            break
    else:
        return None
    left = left.strip()
    if left.lower() == "begin":
        start: datetime | None = None
    else:
        start = parse_datetime_input(left)
        if start is None:
            return None
    right = right.strip()
    if right.lower() in ("", "now"):
        return start, None
    end = parse_datetime_input(right)
    if end is None:
        return None
    return start, end


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


def raw_file_size() -> int:
    """Return the size in bytes of the raw detection log (0 if absent)."""
    try:
        return os.path.getsize(RAW_PATH)
    except OSError:
        return 0


def append_raw_samples(rows: list[tuple]) -> None:
    """Append raw detection samples to the raw CSV log (best-effort).

    Writes the column-label header row the first time the file is created.
    """
    if not rows:
        return
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        write_header = not os.path.exists(RAW_PATH) or os.path.getsize(RAW_PATH) == 0
        with open(RAW_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(RAW_HEADER)
            writer.writerows(rows)
    except OSError:
        pass


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


# Hourly activity chart: a small text bar chart of app usage per hour of day.
# The x axis ticks at 0, 3, 6, …, 24; each column is one hour (0–23).
ACTIVITY_AXIS = "0  3  6  9  12 15 18 21 24"
ACTIVITY_BLOCKS = " ▁▂▃▄▅▆▇█"  # 0 (empty) … 8 (full) eighths


def render_activity_sparkline(hourly: list[float], width: int = 24) -> str:
    """Render per-hour usage seconds as a single monospaced sparkline row.

    The 24 hour-of-day buckets become 24 columns; bar heights are scaled to the
    busiest hour. Empty hours render as a blank column.
    """
    values = [(hourly[i] if i < len(hourly) else 0.0) for i in range(width)]
    peak = max(values) if values else 0.0
    if peak <= 0:
        return " " * width
    cells = []
    for v in values:
        if v <= 0:
            cells.append(" ")
            continue
        level = max(1, min(8, int(round(v / peak * 8))))
        cells.append(ACTIVITY_BLOCKS[level])
    return "".join(cells)


# Emotion heatmap: per-emotion (rows) × hour-of-day (columns) usage intensity.
# Cells shade from empty → light → solid based on duration within the hour.
HEAT_SHADES = " ░▒▓█"  # 0 (empty) … 4 (solid)


def render_emotion_heatmap(
    heat: dict[str, list[float]], keys: list[str], width: int = 24
) -> list[str]:
    """Render an emotion×hour heatmap as one shaded row string per key.

    Intensities are scaled to the busiest single cell across the whole map so
    shades are comparable between emotions. Empty cells render as blanks.
    """
    peak = max((max(row, default=0.0) for row in heat.values()), default=0.0)
    rows = []
    for key in keys:
        hours = heat.get(key, [])
        cells = []
        for i in range(width):
            v = hours[i] if i < len(hours) else 0.0
            if v <= 0 or peak <= 0:
                cells.append(HEAT_SHADES[0])
            else:
                level = max(1, min(4, int(round(v / peak * 4))))
                cells.append(HEAT_SHADES[level])
        rows.append("".join(cells))
    return rows


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


def set_symbol_icon(item, symbol_name: str) -> None:
    """Give a menu item a monochrome SF Symbol icon (template image).

    Template images render in a single colour that automatically adapts to the
    menu's light/dark appearance, so the icons stay monochrome. Best-effort: if
    AppKit/SF Symbols are unavailable (non-macOS or older macOS), the item is
    left unchanged.
    """
    try:
        from AppKit import NSImage

        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            symbol_name, None
        )
        if image is None:
            return
        image.setTemplate_(True)
        item._menuitem.setImage_(image)
    except Exception:  # noqa: BLE001 - optional AppKit dependency
        pass


# Cache of rendered emoji/symbol images so each is drawn only once.
_EMOJI_IMAGE_CACHE: dict[str, object] = {}


def set_emoji_icon(item, text: str, size: float = 14.0) -> None:
    """Render a short string (emoji) to a small colour image used as the item's
    icon, so it sits in the menu's image gutter and the title stays aligned.

    Using the image gutter (instead of putting the emoji in the title text)
    keeps every row's title at the same x regardless of emoji glyph width.
    Best-effort: a no-op if AppKit is unavailable.
    """
    if not text:
        return
    try:
        cached = _EMOJI_IMAGE_CACHE.get(text)
        if cached is None:
            from AppKit import NSFont, NSFontAttributeName, NSImage
            from Foundation import NSAttributedString

            font = NSFont.systemFontOfSize_(size)
            attributed = NSAttributedString.alloc().initWithString_attributes_(
                text, {NSFontAttributeName: font}
            )
            measured = attributed.size()
            width = max(1.0, float(measured.width))
            height = max(1.0, float(measured.height))
            image = NSImage.alloc().initWithSize_((width, height))
            image.lockFocus()
            attributed.drawAtPoint_((0.0, 0.0))
            image.unlockFocus()
            _EMOJI_IMAGE_CACHE[text] = image
            cached = image
        item._menuitem.setImage_(cached)
    except Exception:  # noqa: BLE001 - optional AppKit dependency
        pass


def set_spacer_icon(item, width: float = 19.0, height: float = 14.0) -> None:
    """Give a menu item a transparent image so its title aligns in the image
    gutter with rows that have a real emoji/symbol icon. Best-effort no-op."""
    try:
        from AppKit import NSImage

        image = NSImage.alloc().initWithSize_((width, height))
        item._menuitem.setImage_(image)
    except Exception:  # noqa: BLE001 - optional AppKit dependency
        pass


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
        self._sensitivity: dict[str, str] = {}

    @property
    def result(self) -> EmotionResult:
        with self._lock:
            return self._result

    @property
    def sensitivity(self) -> dict[str, str]:
        """Per-emotion detection sensitivity used for inference."""
        with self._lock:
            return dict(self._sensitivity)

    @sensitivity.setter
    def sensitivity(self, value: dict[str, str]) -> None:
        with self._lock:
            self._sensitivity = dict(value)

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
                self._set_result(infer_emotion(scores, self.sensitivity))
            else:
                self._set_result(EmotionResult("no face", 0.0))

            self._stop.wait(SAMPLE_INTERVAL)


class MooditoApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Moodito", title="😐 neutral", quit_button=None)
        self._worker = FaceWorker()
        self._paused = False
        # Menu bar display options, restored from persisted settings:
        #   show_emojis — show the emotion emoji; when off, show the Moodito icon
        #   show_labels — show the emotion label text
        # Legacy "icon only" mode maps to both options being off.
        self._settings = load_settings()
        legacy_icon_only = bool(self._settings.get("icon_only", False))
        self._show_emojis = bool(self._settings.get("show_emojis", not legacy_icon_only))
        self._show_labels = bool(self._settings.get("show_labels", not legacy_icon_only))
        # Per-emotion detection sensitivity, restored from settings and shared
        # with the worker thread that runs inference.
        self._sensitivity = self._load_sensitivity()
        self._worker.sensitivity = self._sensitivity
        # The statistics range always starts pinned to the live, sliding
        # last-24-hours window (the manual range is not persisted across runs).
        self._stats_live_24h = True
        # Last (icon_path, title) applied to the menu bar; skips redundant
        # updates so the status item doesn't flicker every refresh tick.
        self._last_render: tuple[str | None, str | None] | None = None
        self._icon_path = resource_path(MENUBAR_ICON)
        # Persisted per-emotion usage statistics.
        self._stats, self._stats_started_at = load_stats()
        if self._stats_started_at is None:
            # First run: record when tracking began.
            self._stats_started_at = datetime.now().isoformat(timespec="seconds")
            save_stats(self._stats, self._stats_started_at)
        self._last_emotion: str | None = None
        self._ticks_since_save = 0
        # Buffered raw detection samples awaiting flush to RAW_PATH.
        self._raw_buffer: list[tuple] = []
        # Statistics are shown for a user-selectable datetime range. The end may
        # be None, meaning "now" (a live window that keeps updating). The range
        # defaults to the last 24 hours (clamped to when tracking began).
        self._stats_range_start: datetime = datetime.now()
        self._stats_range_end: datetime | None = None
        self._range_stats: dict[str, dict] = {
            key: {"seconds": 0.0, "count": 0} for key in STAT_KEYS
        }
        self._range_last_state: str | None = None
        # App-usage seconds bucketed by hour of day (0–23) over the range.
        self._hourly_activity: list[float] = [0.0] * 24
        # Per-emotion usage seconds bucketed by hour of day, over the range.
        self._hourly_emotion: dict[str, list[float]] = {
            key: [0.0] * 24 for key in STAT_KEYS
        }
        self._set_default_range()
        # Persisted Lemon Squeezy license activation (if any).
        self._license = load_license()
        self._license_active = is_license_active(self._license)
        # Guards the license fields above, which are read on the main thread
        # but mutated by background activate/deactivate/re-check threads.
        self._license_lock = threading.Lock()
        # Set while a license network call is in flight (prevents double taps).
        self._license_busy = threading.Event()
        # Pending alert text to show on the main thread after a network call.
        self._license_alert: str | None = None
        # Set by a background license thread when the menu's license visibility
        # (and any pending alert) need to be applied on the main thread.
        self._license_dirty = threading.Event()

        # Build the live Insights submenu (charts + per-emotion table). The
        # range controls (Since/Range/Last 24 Hours) and the Download/Erase
        # actions live at the top level, around the Insights submenu.
        self._stats_menu = rumps.MenuItem("Insights")
        # Non-clickable label showing when tracking began.
        self._stats_since_item = rumps.MenuItem("Since …", callback=None)
        # Single control to pick the start and end of the window the statistics
        # are computed over (defaults to the last 24 hours).
        self._stats_range_item = rumps.MenuItem(
            "Range …", callback=self.set_stats_range
        )
        # Toggle: when on, the range is pinned to a live, sliding last-24-hours
        # window and the manual Range control is locked; when off, the user can
        # edit the Range freely.
        self._stats_live_item = rumps.MenuItem(
            "Last 24 Hours", callback=self.toggle_live_24h
        )
        self._stats_live_item.state = self._stats_live_24h
        # Hourly activity chart: usage per hour of day, x axis 0–24.
        self._stats_activity_header_item = rumps.MenuItem(
            "Activity", callback=None
        )
        self._stats_menu.add(self._stats_activity_header_item)
        self._stats_activity_item = rumps.MenuItem("activity", callback=None)
        self._stats_menu.add(self._stats_activity_item)
        self._stats_activity_axis_item = rumps.MenuItem("axis", callback=None)
        self._stats_menu.add(self._stats_activity_axis_item)
        self._stats_menu.add(None)
        # Emotion heatmap: one row per emotion, columns are hours of day.
        self._stats_heatmap_header_item = rumps.MenuItem(
            "Heatmap", callback=None
        )
        self._stats_menu.add(self._stats_heatmap_header_item)
        self._stats_heatmap_items: dict[str, rumps.MenuItem] = {}
        for key in STAT_KEYS:
            item = rumps.MenuItem(f"heat-{key}", callback=None)
            set_emoji_icon(item, STAT_EMOJI.get(key, ""))
            self._stats_menu.add(item)
            self._stats_heatmap_items[key] = item
        self._stats_heatmap_axis_item = rumps.MenuItem("heataxis", callback=None)
        self._stats_menu.add(self._stats_heatmap_axis_item)
        self._stats_menu.add(None)
        # Section label for the per-emotion breakdown table.
        self._stats_emotions_header_item = rumps.MenuItem(
            "Statistics", callback=None
        )
        self._stats_menu.add(self._stats_emotions_header_item)
        self._stats_header_item = rumps.MenuItem("Header", callback=None)
        self._stats_menu.add(self._stats_header_item)
        self._stats_items: dict[str, rumps.MenuItem] = {}
        for key in STAT_KEYS:
            item = rumps.MenuItem(key, callback=None)
            set_emoji_icon(item, STAT_EMOJI.get(key, ""))
            self._stats_menu.add(item)
            self._stats_items[key] = item
        self._stats_menu.add(None)
        self._stats_total_item = rumps.MenuItem("Total", callback=None)
        self._stats_menu.add(self._stats_total_item)
        # Download/Erase actions (added to the top level, after Insights).
        self._stats_export_item = rumps.MenuItem(
            "Download (csv)", callback=self.export_csv
        )
        self._stats_reset_item = rumps.MenuItem(
            "Erase", callback=self.reset_stats
        )

        # Buy Me a Coffee submenu: an "open page" action plus the QR code image.
        self._bmc_menu = rumps.MenuItem("Buy Me a Coffee")
        self._bmc_open_item = rumps.MenuItem(
            "Open buymeacoffee.com", callback=self.buy_me_a_coffee
        )
        self._bmc_menu.add(self._bmc_open_item)
        self._bmc_menu.add(
            rumps.MenuItem(
                "",
                icon=resource_path(BMC_QR),
                dimensions=[180, 180],
                callback=self.buy_me_a_coffee,
            )
        )

        # License submenu (Lemon Squeezy). "Activate" is shown only while
        # unlicensed; "Deactivate" only while licensed.
        self._license_menu = rumps.MenuItem("License")
        self._license_status_item = rumps.MenuItem("Status: …", callback=None)
        self._license_menu.add(self._license_status_item)
        # License details, shown only while licensed.
        self._license_key_item = rumps.MenuItem("Key: …", callback=None)
        self._license_menu.add(self._license_key_item)
        self._license_device_item = rumps.MenuItem("Device: …", callback=None)
        self._license_menu.add(self._license_device_item)
        self._license_menu.add(None)
        self._license_activate_item = rumps.MenuItem(
            "Activate License…", callback=self.activate_license_dialog
        )
        self._license_menu.add(self._license_activate_item)
        self._license_deactivate_item = rumps.MenuItem(
            "Deactivate License", callback=self.deactivate_license_action
        )
        self._license_menu.add(self._license_deactivate_item)
        self._license_restore_item = rumps.MenuItem(
            "Manage License…", callback=self.restore_license
        )
        self._license_menu.add(self._license_restore_item)
        self._license_buy_item = rumps.MenuItem(
            "Buy License…", callback=self.buy_license
        )
        self._license_menu.add(self._license_buy_item)

        # Sensitivity submenu: grouped by emotion. Each group leads with a
        # disabled "<emoji> <Emotion>" header, followed by the Low/Normal/High
        # rows which line up in a vertical column; a checkmark marks the active
        # level and one click sets it. Level rows are added with their unique
        # full titles so rumps (which keys items by title and drops duplicates)
        # keeps them all, then their display text is set to the bare level.
        # Dividers separate the groups — no nested submenu and no dialog window.
        self._sensitivity_menu = rumps.MenuItem("Sensitivity")
        self._sensitivity_items: dict[tuple[str, str], rumps.MenuItem] = {}
        for index, emotion in enumerate(SENSITIVITY_EMOTIONS):
            if index:
                self._sensitivity_menu.add(None)
            emoji = EMOTION_EMOJI.get(emotion, "")
            self._sensitivity_menu.add(
                rumps.MenuItem(f"{emoji} {emotion.capitalize()}".strip(), callback=None)
            )
            for level in SENSITIVITY_LEVELS:
                item = rumps.MenuItem(
                    self._sensitivity_full_title(emotion, level),
                    callback=self.set_sensitivity,
                )
                self._sensitivity_menu.add(item)
                item.title = level.capitalize()
                self._sensitivity_items[(emotion, level)] = item
            self._update_sensitivity_states(emotion)

        self.menu = [
            rumps.MenuItem("Detected: …", callback=None),
            None,
            self._stats_since_item,
            self._stats_range_item,
            self._stats_live_item,
            self._stats_menu,
            self._stats_export_item,
            self._stats_reset_item,
            None,
            rumps.MenuItem("Show Emojis", callback=self.toggle_emojis),
            rumps.MenuItem("Show Labels", callback=self.toggle_labels),
            rumps.MenuItem("Camera Grant Access", callback=self.grant_camera),
            self._sensitivity_menu,
            rumps.MenuItem("Pause", callback=self.toggle_pause),
            None,
            self._license_menu,
            self._bmc_menu,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]
        self._detected_item = self.menu["Detected: …"]
        self._pause_item = self.menu["Pause"]
        self._emojis_item = self.menu["Show Emojis"]
        self._labels_item = self.menu["Show Labels"]
        self._camera_item = self.menu["Camera Grant Access"]
        self._quit_item = self.menu["Quit"]
        # Give each actionable menu option a monochrome SF Symbol icon.
        set_symbol_icon(self._detected_item, "magnifyingglass")
        set_symbol_icon(self._stats_menu, "lightbulb")
        set_symbol_icon(self._stats_since_item, "clock")
        set_symbol_icon(self._stats_range_item, "calendar")
        set_symbol_icon(self._stats_live_item, "clock.arrow.circlepath")
        set_symbol_icon(self._stats_activity_header_item, "chart.bar.xaxis")
        set_symbol_icon(self._stats_heatmap_header_item, "square.grid.3x3.fill")
        set_symbol_icon(self._stats_emotions_header_item, "chart.bar")
        set_symbol_icon(self._stats_total_item, "sum")
        # Transparent spacers so the imageless table/chart rows align in the
        # image gutter with the emoji/symbol rows.
        set_spacer_icon(self._stats_header_item)
        set_emoji_icon(self._stats_activity_item, "⏳")
        set_spacer_icon(self._stats_activity_axis_item)
        set_spacer_icon(self._stats_heatmap_axis_item)
        set_symbol_icon(self._stats_export_item, "square.and.arrow.down")
        set_symbol_icon(self._stats_reset_item, "trash")
        set_symbol_icon(self._emojis_item, "face.smiling")
        set_symbol_icon(self._labels_item, "textformat")
        set_symbol_icon(self._camera_item, "camera")
        set_symbol_icon(self._sensitivity_menu, "slider.horizontal.3")
        set_symbol_icon(self._pause_item, "pause.fill")
        set_symbol_icon(self._bmc_menu, "cup.and.saucer.fill")
        set_symbol_icon(self._bmc_open_item, "globe")
        set_symbol_icon(self._license_menu, "key.fill")
        set_symbol_icon(self._license_buy_item, "cart")
        set_symbol_icon(self._license_restore_item, "arrow.clockwise")
        set_symbol_icon(self._license_activate_item, "checkmark.seal")
        set_symbol_icon(self._license_deactivate_item, "xmark.seal")
        set_symbol_icon(self._quit_item, "power")
        # Reflect the restored display options in the menu items' checkmarks.
        self._emojis_item.state = self._show_emojis
        self._labels_item.state = self._show_labels
        self._stats_live_item.state = self._stats_live_24h
        # Lock the manual Range control when the live last-24-hours toggle is on.
        if self._stats_live_24h:
            self._set_default_range()
        self._apply_range_lock()
        self._recompute_range_stats()
        self._update_stats_menu()
        self._apply_license_visibility()

        self._worker.start()
        # If a license is stored, confirm it is still active in the background
        # and fall back to the unlicensed UI if Lemon Squeezy says otherwise.
        if self._license_active:
            threading.Thread(target=self._recheck_license, daemon=True).start()

    def _set_menubar(self, icon_path: str | None, title: str | None) -> None:
        """Apply an icon and/or title to the menu bar, skipping no-op updates.

        Setting the new value(s) before clearing the other avoids a momentary
        empty state where rumps falls back to showing the app name. Repeated
        identical renders are skipped so the status item does not flicker.
        """
        if (icon_path, title) == self._last_render:
            return
        self._last_render = (icon_path, title)
        if icon_path is not None:
            self.icon = icon_path
            self.title = title
        else:
            self.title = title
            self.icon = None

    def _render_emotion(self, result: EmotionResult) -> None:
        """Render an emotion in the menu bar per the show emojis/labels options.

        With emojis off, the Moodito icon replaces the emoji glyph; labels add
        the emotion name. If both options are off, only the Moodito icon shows.
        """
        if self._show_emojis:
            title = (
                f"{result.emoji} {result.label}" if self._show_labels else result.emoji
            )
            self._set_menubar(None, title)
        else:
            label = result.label if self._show_labels else None
            self._set_menubar(self._icon_path, label)

    @rumps.timer(UI_REFRESH_INTERVAL)
    def refresh(self, _timer) -> None:
        # Hide the grant-access item once camera permission is authorized.
        self._camera_item.hidden = camera_authorization_status() == 3

        # Apply any license state change made by a background license thread.
        self._consume_license_updates()

        if self._paused:
            self._accumulate_stats("paused")
            return

        error = self._worker.error
        if error:
            self._set_menubar(None, "⚠️ Moodito")
            self._detected_item.title = f"Detected: error ({error})"
            self._accumulate_stats("error")
            return

        result = self._worker.result
        self._render_emotion(result)
        self._detected_item.title = f"Detected: {result.label} ({result.score:.0%})"
        self._accumulate_stats(result.label, result.score)

    def _accumulate_stats(self, label: str, score: float | None = None) -> None:
        """Add elapsed time to the current emotion and persist periodically."""
        # Record the raw per-sample reading for the raw data export.
        now = datetime.now()
        timestamp = now.isoformat(timespec="milliseconds")
        self._raw_buffer.append(
            (timestamp, label, "" if score is None else f"{score:.4f}")
        )
        if label in self._stats:
            self._stats[label]["seconds"] += UI_REFRESH_INTERVAL
            # Count a new occurrence each time the emotion changes.
            if label != self._last_emotion:
                self._stats[label]["count"] += 1
        self._last_emotion = label
        # Keep the displayed range stats current without re-reading the raw log
        # every tick. When the live last-24-hours toggle is on, pin the window
        # to the last 24 hours; otherwise only a live ("now") window updates.
        if self._stats_live_24h:
            self._set_default_range()
            self._add_to_range_stats(label)
            self._hourly_activity[now.hour] += UI_REFRESH_INTERVAL
            if label in self._hourly_emotion:
                self._hourly_emotion[label][now.hour] += UI_REFRESH_INTERVAL
        elif self._stats_range_end is None and now >= self._stats_range_start:
            self._add_to_range_stats(label)
            self._hourly_activity[now.hour] += UI_REFRESH_INTERVAL
            if label in self._hourly_emotion:
                self._hourly_emotion[label][now.hour] += UI_REFRESH_INTERVAL
        self._update_stats_menu()

        # Flush to disk roughly every 10 seconds to limit write frequency.
        self._ticks_since_save += 1
        if self._ticks_since_save * UI_REFRESH_INTERVAL >= 10:
            self._ticks_since_save = 0
            save_stats(self._stats, self._stats_started_at)
            append_raw_samples(self._raw_buffer)
            self._raw_buffer.clear()
            if self._stats_live_24h:
                # Re-aggregate so samples older than 24h drop out of the window.
                self._recompute_range_stats()

    def _add_to_range_stats(self, label: str) -> None:
        """Add one sample tick to the range stats, counting state changes."""
        if label not in self._range_stats:
            self._range_last_state = label
            return
        self._range_stats[label]["seconds"] += UI_REFRESH_INTERVAL
        if label != self._range_last_state:
            self._range_stats[label]["count"] += 1
        self._range_last_state = label

    def _iter_raw_states(self):
        """Yield (timestamp, state) pairs from the raw log and pending buffer."""
        try:
            with open(RAW_PATH, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader, None)  # skip the header row
                for row in reader:
                    if len(row) < 2:
                        continue
                    ts = parse_iso_datetime(row[0])
                    if ts is not None:
                        yield ts, row[1]
        except OSError:
            pass
        for row in self._raw_buffer:
            ts = parse_iso_datetime(row[0])
            if ts is not None:
                yield ts, row[1]

    def _recompute_range_stats(self) -> None:
        """Aggregate the raw log over the selected range into ``_range_stats``."""
        start = self._stats_range_start
        end = self._stats_range_end if self._stats_range_end is not None else datetime.now()
        stats = {key: {"seconds": 0.0, "count": 0} for key in STAT_KEYS}
        hourly = [0.0] * 24
        heat = {key: [0.0] * 24 for key in STAT_KEYS}
        previous: str | None = None
        for ts, state in self._iter_raw_states():
            if ts < start or ts > end:
                continue
            # Count every sample as app-usage time for the hourly activity chart.
            hourly[ts.hour] += UI_REFRESH_INTERVAL
            if state in stats:
                stats[state]["seconds"] += UI_REFRESH_INTERVAL
                heat[state][ts.hour] += UI_REFRESH_INTERVAL
                if state != previous:
                    stats[state]["count"] += 1
            previous = state
        self._range_stats = stats
        self._hourly_activity = hourly
        self._hourly_emotion = heat
        self._range_last_state = previous

    def _set_default_range(self) -> None:
        """Reset the statistics window to the last 24 hours (live, ending now)."""
        now = datetime.now()
        tracking_start = parse_iso_datetime(self._stats_started_at) or now
        start = now - timedelta(hours=24)
        if start < tracking_start:
            start = tracking_start
        self._stats_range_start = start
        self._stats_range_end = None


    def _update_stats_menu(self) -> None:
        """Refresh the Statistics submenu rows as an aligned table."""
        # Emojis live in the menu image gutter (set once at build time), so the
        # titles are plain monospaced text and every row's columns line up.
        header = (
            f"{'Emotion':<9}"
            f"{'%':>5}"
            f"{'Time':>9}"
            f"{'Count':>7}"
        )
        set_monospaced_title(self._stats_header_item, header)
        total = sum(entry["seconds"] for entry in self._range_stats.values())
        total_count = sum(entry["count"] for entry in self._range_stats.values())
        for key, item in self._stats_items.items():
            entry = self._range_stats[key]
            pct = (entry["seconds"] / total * 100.0) if total else 0.0
            # Fixed-width columns: name | percent | duration | count.
            row = (
                f"{key:<9}"
                f"{pct:>4.0f}%"
                f"{format_duration(entry['seconds']):>9}"
                f"{'×' + str(entry['count']):>7}"
            )
            set_monospaced_title(item, row)
        # Totals row (Σ shown as the item's icon), same columns as above.
        total_row = (
            f"{'Total':<9}"
            f"{100 if total else 0:>4.0f}%"
            f"{format_duration(total):>9}"
            f"{'×' + str(total_count):>7}"
        )
        set_monospaced_title(self._stats_total_item, total_row)
        # "Since" shows when data collection began (not the range start),
        # alongside the raw data size.
        self._stats_since_item.title = (
            f"Since {format_timestamp(self._stats_started_at)}"
            f" · {format_bytes(raw_file_size())}"
        )
        end_text = (
            "Now"
            if self._stats_range_end is None
            else format_datetime(self._stats_range_end)
        )
        self._stats_range_item.title = (
            f"Range: {format_datetime(self._stats_range_start)} → {end_text}"
        )
        self._stats_reset_item.title = "Erase"
        self._update_activity_chart()
        self._update_emotion_heatmap()

    def _update_activity_chart(self) -> None:
        """Refresh the hourly activity sparkline and axis labels."""
        peak = max(self._hourly_activity) if self._hourly_activity else 0.0
        self._stats_activity_header_item.title = (
            f"Activity · busiest hour {format_duration(peak)}"
        )
        set_monospaced_title(
            self._stats_activity_item,
            render_activity_sparkline(self._hourly_activity),
        )
        set_monospaced_title(self._stats_activity_axis_item, ACTIVITY_AXIS)

    def _update_emotion_heatmap(self) -> None:
        """Refresh the emotion×hour heatmap rows and axis labels."""
        strips = render_emotion_heatmap(self._hourly_emotion, STAT_KEYS)
        # Emojis are the rows' icons (set at build time), so the strips are
        # plain monospaced titles that all start at the same gutter x.
        for key, strip in zip(STAT_KEYS, strips):
            set_monospaced_title(self._stats_heatmap_items[key], strip)
        set_monospaced_title(self._stats_heatmap_axis_item, ACTIVITY_AXIS)


    def set_stats_range(self, _sender) -> None:
        """Prompt for the statistics date/time range and apply it.

        The start cannot be earlier than the "Since" (tracking) start, the end
        cannot be later than now, and the start must be strictly before the end;
        an end of 'now' keeps the window live.
        """
        # The range is locked while the live last-24-hours window is on.
        if self._stats_live_24h:
            return
        now = datetime.now()
        tracking_start = parse_iso_datetime(self._stats_started_at) or now
        # Pre-fill the prompt with the last applied range so the user's
        # previously entered datetimes are kept.
        start = self._stats_range_start
        end_text = (
            "now"
            if self._stats_range_end is None
            else self._stats_range_end.strftime("%Y-%m-%d %H:%M")
        )
        window = rumps.Window(
            title=STATS_RANGE_TITLE,
            message=(
                "Datetime format YYYY-MM-DD HH:MM\n\n"
                "Use 'begin' as the start for the Since datetime, "
                "and 'now' as the end for a live window."
            ),
            default_text=f"{start.strftime('%Y-%m-%d %H:%M')} to {end_text}",
            ok="Apply",
            cancel="Cancel",
            dimensions=(260, 24),
        )
        response = window.run()
        if not response.clicked:
            return
        parsed = parse_datetime_range(response.text)
        if parsed is None:
            rumps.alert(
                STATS_RANGE_TITLE,
                "Could not understand that range.\n"
                "Use the format START to END (YYYY-MM-DD HH:MM).",
            )
            return
        start_val, end_val = parsed
        # A 'begin' start resolves to the "Since" (data start) datetime.
        if start_val is None:
            start_val = tracking_start
        # The start cannot be before the "Since" (tracking) start.
        start_val = max(start_val, tracking_start)
        if end_val is not None:
            # The end cannot be after now.
            end_val = min(end_val, now)
        # The start cannot equal or come after the end. A live end ("now")
        # is treated as the current time for this comparison.
        effective_end = end_val if end_val is not None else now
        if start_val >= effective_end:
            rumps.alert(
                STATS_RANGE_TITLE,
                "The start must be before the end.",
            )
            return
        self._stats_range_start = start_val
        self._stats_range_end = end_val
        self._recompute_range_stats()
        self._update_stats_menu()

    def toggle_live_24h(self, sender) -> None:
        """Toggle the live last-24-hours window on or off.

        When on, the range is pinned to a sliding last-24-hours window and the
        manual Range control is locked. When off, the Range becomes editable
        and keeps whatever window was last shown. Turning it off (to use a
        custom range) requires an active license.
        """
        # Turning the live window off unlocks custom date ranges — a licensed
        # feature. Block it while unlicensed and keep the option checked.
        if self._stats_live_24h and not self._license_active:
            sender.state = True
            rumps.alert(
                LICENSE_ALERT_TITLE,
                "Custom date ranges are a licensed feature.\n\n"
                "Activate a license to turn off “Last 24 Hours” and choose "
                "your own range.",
            )
            return
        self._stats_live_24h = not self._stats_live_24h
        sender.state = self._stats_live_24h
        if self._stats_live_24h:
            self._set_default_range()
        self._apply_range_lock()
        self._recompute_range_stats()
        self._update_stats_menu()

    def _apply_range_lock(self) -> None:
        """Enable or disable the manual Range control based on the live toggle."""
        self._stats_range_item.set_callback(
            None if self._stats_live_24h else self.set_stats_range
        )

    def toggle_emojis(self, sender) -> None:
        self._show_emojis = not self._show_emojis
        sender.state = self._show_emojis
        # Persist the choice so it is restored on the next launch.
        self._settings["show_emojis"] = self._show_emojis
        save_settings(self._settings)
        # Apply immediately rather than waiting for the next refresh tick.
        if not self._paused:
            self.refresh(None)

    def toggle_labels(self, sender) -> None:
        self._show_labels = not self._show_labels
        sender.state = self._show_labels
        # Persist the choice so it is restored on the next launch.
        self._settings["show_labels"] = self._show_labels
        save_settings(self._settings)
        # Apply immediately rather than waiting for the next refresh tick.
        if not self._paused:
            self.refresh(None)

    def _load_sensitivity(self) -> dict[str, str]:
        """Return the per-emotion sensitivity, restored from settings.

        Unknown emotions/levels fall back to the default so an edited or stale
        settings file can never break inference.
        """
        stored = self._settings.get("sensitivity")
        result = dict.fromkeys(SENSITIVITY_EMOTIONS, DEFAULT_SENSITIVITY)
        if isinstance(stored, dict):
            for emotion in SENSITIVITY_EMOTIONS:
                level = stored.get(emotion)
                if level in SENSITIVITY_LEVELS:
                    result[emotion] = level
        return result

    def _sensitivity_full_title(self, emotion: str, level: str) -> str:
        """Unique add-time row title (only used as the rumps menu key)."""
        emoji = EMOTION_EMOJI.get(emotion, "")
        return f"{level.capitalize()} · {emoji} {emotion.capitalize()}".strip()

    def _update_sensitivity_states(self, emotion: str) -> None:
        """Move the radio checkmark to the emotion's selected level."""
        selected = self._sensitivity.get(emotion, DEFAULT_SENSITIVITY)
        for level in SENSITIVITY_LEVELS:
            self._sensitivity_items[(emotion, level)].state = level == selected

    def set_sensitivity(self, sender) -> None:
        """Apply the level chosen from the Sensitivity menu."""
        target = next(
            (key for key, item in self._sensitivity_items.items() if item is sender),
            None,
        )
        if target is None:
            return
        emotion, level = target
        self._sensitivity[emotion] = level
        self._update_sensitivity_states(emotion)
        # Share the new setting with the inference thread and persist it.
        self._worker.sensitivity = self._sensitivity
        self._settings["sensitivity"] = self._sensitivity
        save_settings(self._settings)

    def toggle_pause(self, _sender) -> None:
        self._paused = not self._paused
        if self._paused:
            self._pause_item.title = "Resume"
            set_symbol_icon(self._pause_item, "play.fill")
            self._set_menubar(None, "⏸️ Moodito")
            self._detected_item.title = "Detected: paused"
        else:
            self._pause_item.title = "Pause"
            set_symbol_icon(self._pause_item, "pause.fill")

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

    def _apply_license_visibility(self) -> None:
        """Sync menu visibility and status text with the current license state.

        Activate is shown only while unlicensed; Deactivate only while
        licensed; the Buy Me a Coffee tip jar is hidden once licensed.
        """
        active = self._license_active
        self._license_activate_item.hidden = active
        self._license_deactivate_item.hidden = not active
        # Buy is only useful before purchasing; hide it once licensed.
        self._license_buy_item.hidden = active
        self._bmc_menu.hidden = active
        self._license_status_item.title = (
            "Status: Licensed ✓" if active else "Status: Not licensed"
        )
        # License details: shown only while licensed.
        self._license_key_item.hidden = not active
        self._license_device_item.hidden = not active
        if active:
            with self._license_lock:
                key = self._license.get("license_key", "")
                device = self._license.get("instance_name", "")
            self._license_key_item.title = f"Key: {mask_license_key(key)}"
            self._license_device_item.title = (
                f"Device: {device}" if device else "Device: —"
            )
        # Custom date ranges are licensed: if the license is gone while a custom
        # range is active, re-pin to the live last-24-hours window.
        if not active and getattr(self, "_stats_live_24h", True) is False:
            self._stats_live_24h = True
            self._stats_live_item.state = True
            self._set_default_range()
            self._apply_range_lock()
            self._recompute_range_stats()
            self._update_stats_menu()

    def _consume_license_updates(self) -> None:
        """Apply any pending license state change on the main thread.

        Background license threads only mutate the shared license fields and
        set ``_license_dirty``; the actual menu update and any user-facing
        alert happen here, on the main (UI) thread.
        """
        if not self._license_dirty.is_set():
            return
        self._license_dirty.clear()
        self._apply_license_visibility()
        with self._license_lock:
            alert = self._license_alert
            self._license_alert = None
        if alert:
            rumps.alert(LICENSE_ALERT_TITLE, alert)

    @rumps.timer(LICENSE_RECHECK_INTERVAL)
    def _periodic_license_check(self, _timer) -> None:
        """Periodically re-validate an active license while the app runs."""
        if self._license_active and not self._license_busy.is_set():
            threading.Thread(target=self._recheck_license, daemon=True).start()

    def _recheck_license(self) -> None:
        """Background: confirm the stored license is still valid (fallback).

        If Lemon Squeezy reports the license as invalid, the activation is
        cleared locally and the menu falls back to the unlicensed state. A
        network/server error is treated as transient and left untouched.
        """
        with self._license_lock:
            key = self._license.get("license_key", "")
            instance_id = self._license.get("instance_id", "")
        if validate_license(key, instance_id) != LICENSE_INVALID:
            # Valid, or could not reach the server (transient) → keep as-is.
            return
        clear_license()
        with self._license_lock:
            self._license = {}
            self._license_active = False
        self._license_dirty.set()

    def buy_license(self, _sender) -> None:
        """Open the Lemon Squeezy storefront to purchase a license."""
        subprocess.run(["open", LICENSE_BUY_URL], check=False)

    def restore_license(self, _sender) -> None:
        """Open the Lemon Squeezy customer portal to find a past order."""
        subprocess.run(["open", LICENSE_RESTORE_URL], check=False)

    def activate_license_dialog(self, _sender) -> None:
        """Prompt for a license key and activate it (network call off-thread)."""
        if self._license_busy.is_set():
            return
        window = rumps.Window(
            title="Moodito License",
            message="Enter your license key:",
            default_text="",
            ok="Activate",
            cancel="Cancel",
            dimensions=(320, 24),
        )
        response = window.run()
        if response.clicked != 1:
            return
        key = response.text.strip()
        if not key:
            return
        self._license_busy.set()
        self._license_status_item.title = "Status: Activating…"
        threading.Thread(
            target=self._activate_worker, args=(key,), daemon=True
        ).start()

    def _activate_worker(self, key: str) -> None:
        """Background: activate ``key`` and stage the result for the UI thread."""
        instance_name = license_instance_name()
        ok, message, instance_id = activate_license(key, instance_name)
        if ok:
            license_data = {
                "license_key": key,
                "instance_id": instance_id,
                "instance_name": instance_name,
            }
            save_license(license_data)
            with self._license_lock:
                self._license = license_data
                self._license_active = True
                self._license_alert = "License activated. Thank you! 🎉"
        else:
            with self._license_lock:
                self._license_alert = f"Could not activate license:\n{message}"
        self._license_busy.clear()
        self._license_dirty.set()

    def deactivate_license_action(self, _sender) -> None:
        """Deactivate the current license (network call off-thread)."""
        if self._license_busy.is_set():
            return
        with self._license_lock:
            key = self._license.get("license_key", "")
            instance_id = self._license.get("instance_id", "")
        self._license_busy.set()
        self._license_status_item.title = "Status: Deactivating…"
        threading.Thread(
            target=self._deactivate_worker, args=(key, instance_id), daemon=True
        ).start()

    def _deactivate_worker(self, key: str, instance_id: str) -> None:
        """Background: deactivate the license and stage the UI-thread result."""
        ok, message = deactivate_license(key, instance_id)
        if ok:
            clear_license()
            with self._license_lock:
                self._license = {}
                self._license_active = False
                self._license_alert = "License deactivated."
        else:
            with self._license_lock:
                self._license_alert = f"Could not deactivate license:\n{message}"
        self._license_busy.clear()
        self._license_dirty.set()

    def export_csv(self, _sender) -> None:
        """Export the raw detection log for the selected date range to a CSV
        file in the Downloads folder."""
        # Flush any buffered samples first so the export includes everything.
        append_raw_samples(self._raw_buffer)
        self._raw_buffer.clear()
        start = self._stats_range_start
        end = self._stats_range_end if self._stats_range_end is not None else datetime.now()
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        filename = f"moodito-raw-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        path = os.path.join(downloads, filename)
        try:
            with open(path, "w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                writer.writerow(RAW_HEADER)
                if os.path.exists(RAW_PATH):
                    with open(RAW_PATH, newline="", encoding="utf-8") as src:
                        reader = csv.reader(src)
                        next(reader, None)  # skip the header row
                        for row in reader:
                            if not row:
                                continue
                            ts = parse_iso_datetime(row[0])
                            if ts is None or ts < start or ts > end:
                                continue
                            writer.writerow(row)
        except OSError as exc:
            rumps.notification("Moodito", "Export failed", str(exc))
            return
        # Reveal the exported file in Finder.
        subprocess.run(["open", "-R", path], check=False)

    def reset_stats(self, _sender) -> None:
        """Clear all accumulated statistics and the raw detection log."""
        self._stats = {
            key: {"seconds": 0.0, "count": 0} for key in STAT_KEYS
        }
        self._last_emotion = None
        self._raw_buffer.clear()
        try:
            os.remove(RAW_PATH)
        except OSError:
            pass
        self._stats_started_at = datetime.now().isoformat(timespec="seconds")
        save_stats(self._stats, self._stats_started_at)
        self._set_default_range()
        self._recompute_range_stats()
        self._update_stats_menu()

    def quit_app(self, _sender) -> None:
        self._worker.stop()
        save_stats(self._stats, self._stats_started_at)
        append_raw_samples(self._raw_buffer)
        self._raw_buffer.clear()
        rumps.quit_application()


def main() -> None:
    # Validate the emoji table is wired up (cheap sanity check at startup).
    assert "happy" in EMOTION_EMOJI
    # Ask for camera access up front so macOS shows the permission prompt.
    request_camera_access()
    MooditoApp().run()


if __name__ == "__main__":
    main()
