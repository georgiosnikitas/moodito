"""Unit tests for app.py — non-GUI helpers and worker/state logic.

These avoid constructing the full rumps event loop; GUI wiring is exercised
only through standalone helpers and bare-instance method calls.
"""
from __future__ import annotations

import builtins
import json

import pytest

import app
from emotion import EmotionResult


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point all persistence paths at a temporary directory."""
    monkeypatch.setattr(app, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(app, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(app, "STATS_PATH", str(tmp_path / "stats.json"))
    monkeypatch.setattr(app, "RAW_PATH", str(tmp_path / "raw_data.csv"))
    monkeypatch.setattr(app, "MODEL_PATH", str(tmp_path / "model.task"))
    return tmp_path


class TestResourcePath:
    def test_uses_meipass_when_present(self, monkeypatch) -> None:
        monkeypatch.setattr(app.sys, "_MEIPASS", "/bundle", raising=False)
        assert app.resource_path("x.png") == "/bundle/x.png"

    def test_falls_back_to_module_dir(self, monkeypatch) -> None:
        monkeypatch.delattr(app.sys, "_MEIPASS", raising=False)
        result = app.resource_path("x.png")
        assert result.endswith("x.png")
        assert "/bundle/" not in result


class TestEnsureModel:
    def test_skips_download_when_model_exists(self, data_dir, monkeypatch) -> None:
        (data_dir / "model.task").write_bytes(b"data")
        called = False

        def fake_urlretrieve(*_a, **_k):
            nonlocal called
            called = True

        monkeypatch.setattr(app.urllib.request, "urlretrieve", fake_urlretrieve)
        app.ensure_model()
        assert called is False

    def test_downloads_when_model_absent(self, data_dir, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            app.urllib.request,
            "urlretrieve",
            lambda url, path: calls.append((url, path)),
        )
        app.ensure_model()
        assert calls == [(app.MODEL_URL, app.MODEL_PATH)]


class TestSettings:
    def test_load_missing_returns_empty(self, data_dir) -> None:
        assert app.load_settings() == {}

    def test_save_then_load_roundtrip(self, data_dir) -> None:
        app.save_settings({"icon_only": True})
        assert app.load_settings() == {"icon_only": True}

    def test_non_dict_json_returns_empty(self, data_dir) -> None:
        (data_dir / "settings.json").write_text("[1, 2, 3]")
        assert app.load_settings() == {}

    def test_corrupt_json_returns_empty(self, data_dir) -> None:
        (data_dir / "settings.json").write_text("{not valid")
        assert app.load_settings() == {}

    def test_save_swallows_os_error(self, data_dir, monkeypatch) -> None:
        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(app.os, "makedirs", boom)
        # Best-effort: must not raise even when the write fails.
        app.save_settings({"icon_only": True})


class TestStats:
    def test_load_missing_returns_normalised_zeros(self, data_dir) -> None:
        stats, started = app.load_stats()
        assert started is None
        assert set(stats) == set(app.STAT_KEYS)
        assert all(v == {"seconds": 0.0, "count": 0} for v in stats.values())

    def test_save_then_load_roundtrip(self, data_dir) -> None:
        stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
        stats["happy"] = {"seconds": 12.5, "count": 3}
        app.save_stats(stats, "2026-06-21T22:00:00")
        loaded, started = app.load_stats()
        assert started == "2026-06-21T22:00:00"
        assert loaded["happy"] == {"seconds": 12.5, "count": 3}

    def test_legacy_flat_format_is_supported(self, data_dir) -> None:
        # Older format stored emotions at the top level (no wrapper).
        (data_dir / "stats.json").write_text(
            json.dumps({"sad": {"seconds": 5.0, "count": 2}})
        )
        stats, started = app.load_stats()
        assert started is None
        assert stats["sad"] == {"seconds": 5.0, "count": 2}

    def test_corrupt_stats_returns_zeros(self, data_dir) -> None:
        (data_dir / "stats.json").write_text("{broken")
        stats, started = app.load_stats()
        assert started is None
        assert stats["happy"] == {"seconds": 0.0, "count": 0}

    def test_save_swallows_os_error(self, data_dir, monkeypatch) -> None:
        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(app.os, "makedirs", boom)
        app.save_stats({}, None)  # must not raise


class TestFormatTimestamp:
    def test_none_returns_dash(self) -> None:
        assert app.format_timestamp(None) == "—"

    def test_valid_iso_is_formatted(self) -> None:
        assert app.format_timestamp("2026-06-21T22:10:00") == "Jun 21, 2026 · 22:10"

    def test_invalid_string_is_returned_unchanged(self) -> None:
        assert app.format_timestamp("not-a-date") == "not-a-date"


class TestFormatBytes:
    @pytest.mark.parametrize(
        "num,expected",
        [
            (512, "512 B"),
            (1536, "1.5 KB"),
            (1024 * 1024 * 2, "2.0 MB"),
            (1024**3 * 3, "3.0 GB"),
        ],
    )
    def test_formats(self, num, expected) -> None:
        assert app.format_bytes(num) == expected


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (5, "5s"),
            (65, "1m 05s"),
            (3725, "1h 02m"),
        ],
    )
    def test_formats(self, seconds, expected) -> None:
        assert app.format_duration(seconds) == expected


class TestStatsFileSize:
    def test_absent_returns_zero(self, data_dir) -> None:
        assert app.stats_file_size() == 0

    def test_present_returns_size(self, data_dir) -> None:
        (data_dir / "stats.json").write_bytes(b"abcde")
        assert app.stats_file_size() == 5


class TestRawFileSize:
    def test_absent_returns_zero(self, data_dir) -> None:
        assert app.raw_file_size() == 0

    def test_present_returns_size(self, data_dir) -> None:
        (data_dir / "raw_data.csv").write_bytes(b"abcdefg")
        assert app.raw_file_size() == 7


class TestAppendRawSamples:
    def test_empty_rows_writes_nothing(self, data_dir) -> None:
        app.append_raw_samples([])
        assert not (data_dir / "raw_data.csv").exists()

    def test_writes_header_then_rows(self, data_dir) -> None:
        app.append_raw_samples([("2026-06-21T22:00:00.000", "happy", "0.9000")])
        lines = (data_dir / "raw_data.csv").read_text().splitlines()
        assert lines[0] == "timestamp,state,score"
        assert lines[1] == "2026-06-21T22:00:00.000,happy,0.9000"

    def test_appends_without_duplicating_header(self, data_dir) -> None:
        app.append_raw_samples([("t1", "happy", "0.9")])
        app.append_raw_samples([("t2", "sad", "")])
        lines = (data_dir / "raw_data.csv").read_text().splitlines()
        assert lines == [
            "timestamp,state,score",
            "t1,happy,0.9",
            "t2,sad,",
        ]

    def test_swallows_os_error(self, data_dir, monkeypatch) -> None:
        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(app.os, "makedirs", boom)
        app.append_raw_samples([("t", "happy", "0.9")])  # must not raise


class TestCameraAccess:
    def test_status_returns_none_without_avfoundation(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "AVFoundation":
                raise ImportError("no AVFoundation")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert app.camera_authorization_status() is None

    def test_request_access_noop_when_already_decided(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        # Should return without importing AVFoundation or raising.
        app.request_camera_access()

    def test_request_access_handles_missing_avfoundation(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 0)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "AVFoundation":
                raise ImportError("no AVFoundation")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        app.request_camera_access()  # must not raise

    def test_status_returns_int_or_none_live(self) -> None:
        # Exercises the real AVFoundation branch on macOS runners.
        result = app.camera_authorization_status()
        assert result is None or result in (0, 1, 2, 3)


class TestMonospacedTitle:
    def test_success_path_sets_attributed_title(self) -> None:
        import rumps

        item = rumps.MenuItem("row")
        app.set_monospaced_title(item, "hello")  # uses AppKit on macOS

    def test_fallback_sets_plain_title(self, monkeypatch) -> None:
        import rumps

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "AppKit":
                raise ImportError("no AppKit")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        item = rumps.MenuItem("row")
        app.set_monospaced_title(item, "plain text")
        assert item.title == "plain text"


class TestSymbolIcon:
    def test_sets_template_image_on_macos(self) -> None:
        import rumps

        item = rumps.MenuItem("row")
        app.set_symbol_icon(item, "gear")  # valid SF Symbol on macOS

    def test_invalid_symbol_is_noop(self) -> None:
        import rumps

        item = rumps.MenuItem("row")
        # An unknown symbol name returns None; must not raise.
        app.set_symbol_icon(item, "definitely-not-a-real-symbol-xyz")

    def test_fallback_when_appkit_missing(self, monkeypatch) -> None:
        import rumps

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "AppKit":
                raise ImportError("no AppKit")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        item = rumps.MenuItem("row")
        app.set_symbol_icon(item, "gear")  # must not raise


class TestOpenActions:
    def test_open_camera_settings_invokes_open(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: calls.append(a))
        app.open_camera_settings()
        assert calls and calls[0][0][0] == "open"

    def test_buy_me_a_coffee_opens_url(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: calls.append(a))
        # Method does not use `self`; pass a sentinel.
        app.MooditoApp.buy_me_a_coffee(object(), None)
        assert calls[0][0] == ["open", app.BMC_URL]


class TestFaceWorker:
    def test_initial_state(self) -> None:
        worker = app.FaceWorker()
        assert worker.result == EmotionResult("neutral", 0.0)
        assert worker.error is None
        assert worker.ready is False

    def test_set_result_updates_state(self) -> None:
        worker = app.FaceWorker()
        worker._set_result(EmotionResult("happy", 0.9))
        assert worker.result == EmotionResult("happy", 0.9)
        assert worker.ready is True
        assert worker.error is None

    def test_set_error_records_message(self) -> None:
        worker = app.FaceWorker()
        worker._set_error("boom")
        assert worker.error == "boom"

    def test_stop_sets_event(self) -> None:
        worker = app.FaceWorker()
        worker.stop()
        assert worker._stop.is_set()


def _bare_app():
    """Build a MooditoApp instance with only the attributes stats logic needs."""
    import rumps

    inst = object.__new__(app.MooditoApp)
    inst._stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
    inst._last_emotion = None
    inst._ticks_since_save = 0
    inst._raw_buffer = []
    inst._stats_started_at = "2026-06-21T22:00:00"
    inst._stats_items = {k: rumps.MenuItem(k) for k in app.STAT_KEYS}
    inst._stats_header_item = rumps.MenuItem("header")
    inst._stats_total_item = rumps.MenuItem("total")
    inst._stats_since_item = rumps.MenuItem("since")
    inst._stats_reset_item = rumps.MenuItem("reset")
    return inst


class TestStatsAccumulation:
    def test_accumulate_adds_time_and_counts_changes(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        inst = _bare_app()
        inst._accumulate_stats("happy")
        assert inst._stats["happy"]["seconds"] == pytest.approx(app.UI_REFRESH_INTERVAL)
        assert inst._stats["happy"]["count"] == 1
        # Same emotion again: time grows, count stays.
        inst._accumulate_stats("happy")
        assert inst._stats["happy"]["count"] == 1
        # Switching emotion bumps the new one's count.
        inst._accumulate_stats("sad")
        assert inst._stats["sad"]["count"] == 1

    def test_unknown_label_is_ignored(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        inst = _bare_app()
        inst._accumulate_stats("not-a-tracked-state")
        assert all(v["count"] == 0 for v in inst._stats.values())

    def test_periodic_save_flushes(self, monkeypatch) -> None:
        saved = []
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: saved.append(a))
        monkeypatch.setattr(app, "append_raw_samples", lambda *a, **k: None)
        inst = _bare_app()
        ticks = int(10 / app.UI_REFRESH_INTERVAL) + 1
        for _ in range(ticks):
            inst._accumulate_stats("happy")
        assert saved  # at least one periodic flush occurred

    def test_update_stats_menu_sets_titles(self) -> None:
        inst = _bare_app()
        inst._stats["happy"] = {"seconds": 30.0, "count": 2}
        inst._update_stats_menu()
        assert "Since" in inst._stats_since_item.title
        assert "Reset Statistics" in inst._stats_reset_item.title


class TestRawRecording:
    def test_accumulate_buffers_raw_sample_with_score(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        inst = _bare_app()
        inst._accumulate_stats("happy", 0.9123)
        assert len(inst._raw_buffer) == 1
        timestamp, state, score = inst._raw_buffer[0]
        assert state == "happy"
        assert score == "0.9123"
        assert timestamp  # ISO timestamp string

    def test_accumulate_buffers_empty_score_when_none(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        inst = _bare_app()
        inst._accumulate_stats("paused")
        assert inst._raw_buffer[0][1] == "paused"
        assert inst._raw_buffer[0][2] == ""

    def test_periodic_flush_appends_and_clears_buffer(self, monkeypatch) -> None:
        flushed = []
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        monkeypatch.setattr(
            app, "append_raw_samples", lambda rows: flushed.extend(rows)
        )
        inst = _bare_app()
        ticks = int(10 / app.UI_REFRESH_INTERVAL) + 1
        for _ in range(ticks):
            inst._accumulate_stats("happy", 0.5)
        assert flushed  # raw samples were flushed
        assert inst._raw_buffer == []  # buffer cleared after flush


class TestExportCsv:
    def test_copies_existing_raw_file_to_downloads(
        self, data_dir, monkeypatch
    ) -> None:
        # Pretend HOME is the temp dir and create a Downloads folder.
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        (data_dir / "raw_data.csv").write_text("timestamp,state,score\nt,happy,0.9\n")
        opened = []
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: opened.append(a))
        inst = _bare_app()
        inst.export_csv(None)
        exported = list((data_dir / "Downloads").glob("moodito-raw-*.csv"))
        assert len(exported) == 1
        assert "happy" in exported[0].read_text()
        assert opened and opened[0][0][0] == "open"

    def test_flushes_buffer_before_export(self, data_dir, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: None)
        inst = _bare_app()
        inst._raw_buffer = [("t", "happy", "0.9")]
        inst.export_csv(None)
        assert inst._raw_buffer == []
        exported = list((data_dir / "Downloads").glob("moodito-raw-*.csv"))
        assert "happy" in exported[0].read_text()

    def test_writes_header_only_when_no_raw_file(
        self, data_dir, monkeypatch
    ) -> None:
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: None)
        inst = _bare_app()
        inst.export_csv(None)
        exported = list((data_dir / "Downloads").glob("moodito-raw-*.csv"))
        assert exported[0].read_text().strip() == "timestamp,state,score"

    def test_notifies_on_os_error(self, data_dir, monkeypatch) -> None:
        # Missing Downloads folder makes the copy raise OSError.
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "raw_data.csv").write_text("timestamp,state,score\n")
        notes = []
        monkeypatch.setattr(
            app.rumps, "notification", lambda *a, **k: notes.append(a)
        )
        ran = []
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: ran.append(a))
        inst = _bare_app()
        inst.export_csv(None)
        assert notes  # a failure notification was shown
        assert not ran  # Finder reveal was skipped


class TestResetStats:
    def test_clears_stats_and_raw_log(self, data_dir, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        (data_dir / "raw_data.csv").write_text("timestamp,state,score\nt,happy,0.9\n")
        inst = _bare_app()
        inst._stats["happy"] = {"seconds": 30.0, "count": 5}
        inst._raw_buffer = [("t", "happy", "0.9")]
        inst.reset_stats(None)
        assert all(v == {"seconds": 0.0, "count": 0} for v in inst._stats.values())
        assert inst._raw_buffer == []
        assert not (data_dir / "raw_data.csv").exists()
        assert inst._last_emotion is None

    def test_missing_raw_file_is_tolerated(self, data_dir, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        inst = _bare_app()
        inst.reset_stats(None)  # no raw file present; must not raise


class TestQuitApp:
    def test_stops_worker_and_flushes_raw(self, monkeypatch) -> None:
        saved = []
        flushed = []
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: saved.append(a))
        monkeypatch.setattr(
            app, "append_raw_samples", lambda rows: flushed.extend(rows)
        )
        monkeypatch.setattr(app.rumps, "quit_application", lambda: None)
        inst = _bare_app()
        stopped = []
        inst._worker = type("W", (), {"stop": lambda self: stopped.append(True)})()
        inst._raw_buffer = [("t", "happy", "0.9")]
        inst.quit_app(None)
        assert stopped == [True]
        assert flushed == [("t", "happy", "0.9")]
        assert inst._raw_buffer == []
        assert saved


@pytest.fixture
def full_app(data_dir, monkeypatch):
    """Construct a full MooditoApp without starting the camera thread."""
    monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
    monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
    return app.MooditoApp()


class TestMooditoAppInit:
    def test_builds_menu_and_records_start_time(self, full_app) -> None:
        assert full_app._stats_started_at is not None
        assert full_app._raw_buffer == []
        assert "Download Raw Data (CSV)" in full_app._stats_export_item.title
        assert set(full_app._stats_items) == set(app.STAT_KEYS)

    def test_restores_icon_only_setting(self, data_dir, monkeypatch) -> None:
        app.save_settings({"icon_only": True})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._icon_only is True
        assert inst._icon_only_item.state == 1


class TestRefresh:
    def test_normal_updates_detected_and_buffers_raw(self, full_app) -> None:
        full_app._worker._set_result(EmotionResult("happy", 0.8))
        full_app.refresh(None)
        assert "happy" in full_app._detected_item.title
        assert full_app._raw_buffer[-1][1] == "happy"

    def test_paused_records_paused_state(self, full_app) -> None:
        full_app._paused = True
        full_app.refresh(None)
        assert full_app._raw_buffer[-1][1] == "paused"

    def test_error_updates_detected_title(self, full_app) -> None:
        full_app._worker._set_error("camera gone")
        full_app.refresh(None)
        assert "error" in full_app._detected_item.title
        assert full_app._raw_buffer[-1][1] == "error"


class TestRenderStatus:
    def test_icon_mode_shows_icon(self, full_app) -> None:
        full_app._icon_only = True
        full_app._showing_icon = False
        full_app._render_status("😀 happy", allow_icon=True)
        assert full_app._showing_icon is True

    def test_text_mode_sets_title(self, full_app) -> None:
        full_app._showing_icon = True
        full_app._render_status("😀 happy", allow_icon=False)
        assert full_app._showing_icon is False
        assert full_app.title == "😀 happy"


class TestToggles:
    def test_toggle_pause_round_trip(self, full_app) -> None:
        full_app.toggle_pause(None)
        assert full_app._paused is True
        assert "Resume" in full_app._pause_item.title
        full_app.toggle_pause(None)
        assert full_app._paused is False
        assert "Pause" in full_app._pause_item.title

    def test_toggle_icon_only_persists(self, full_app, monkeypatch) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        full_app._paused = True  # skip the refresh branch
        full_app.toggle_icon_only(full_app._icon_only_item)
        assert full_app._icon_only is True
        assert saved and saved[0]["icon_only"] is True


class TestGrantCamera:
    def test_unavailable_shows_alert(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "camera_authorization_status", lambda: None)
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        full_app.grant_camera(None)
        assert alerts

    def test_not_determined_requests_access(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 0)
        called = []
        monkeypatch.setattr(app, "request_camera_access", lambda: called.append(True))
        full_app.grant_camera(None)
        assert called == [True]

    def test_authorized_shows_alert(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        full_app.grant_camera(None)
        assert alerts

    def test_denied_opens_settings(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 2)
        opened = []
        monkeypatch.setattr(app, "open_camera_settings", lambda: opened.append(True))
        full_app.grant_camera(None)
        assert opened == [True]


class _FakeCap:
    """A minimal cv2.VideoCapture stand-in for worker-loop tests."""

    def __init__(self, worker, opened=True, frames=2) -> None:
        self._worker = worker
        self._opened = opened
        self._frames = frames
        self.reads = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        self.reads += 1
        if self.reads >= self._frames:
            self._worker._stop.set()
        return True, "frame"

    def release(self) -> None:
        self.released = True


class _Blendshape:
    category_name = "mouthSmileLeft"
    score = 0.6


class _Detection:
    def __init__(self, has_face=True) -> None:
        self.face_blendshapes = [[_Blendshape()]] if has_face else []


class TestFaceWorkerRun:
    def test_run_reports_model_download_failure(self, monkeypatch) -> None:
        def boom() -> None:
            raise RuntimeError("no network")

        monkeypatch.setattr(app, "ensure_model", boom)
        worker = app.FaceWorker()
        worker.run()
        assert "model download failed" in worker.error

    def test_capture_loop_processes_face_frame(self, monkeypatch) -> None:
        monkeypatch.setattr(app.cv2, "cvtColor", lambda *a, **k: "rgb")
        monkeypatch.setattr(app.cv2, "COLOR_BGR2RGB", 0, raising=False)
        monkeypatch.setattr(app.mp, "Image", lambda **k: "image")
        monkeypatch.setattr(
            app, "infer_emotion", lambda scores: EmotionResult("happy", 0.9)
        )
        worker = app.FaceWorker()
        landmarker = type(
            "L", (), {"detect_for_video": lambda self, img, ts: _Detection(True)}
        )()
        cap = _FakeCap(worker, frames=2)
        worker._run_capture_loop(cap, landmarker)
        assert worker.result == EmotionResult("happy", 0.9)

    def test_capture_loop_handles_no_face(self, monkeypatch) -> None:
        monkeypatch.setattr(app.cv2, "cvtColor", lambda *a, **k: "rgb")
        monkeypatch.setattr(app.cv2, "COLOR_BGR2RGB", 0, raising=False)
        monkeypatch.setattr(app.mp, "Image", lambda **k: "image")
        worker = app.FaceWorker()
        landmarker = type(
            "L", (), {"detect_for_video": lambda self, img, ts: _Detection(False)}
        )()
        cap = _FakeCap(worker, frames=2)
        worker._run_capture_loop(cap, landmarker)
        assert worker.result == EmotionResult("no face", 0.0)

    def test_capture_loop_reopens_after_many_read_failures(self, monkeypatch) -> None:
        worker = app.FaceWorker()

        class FailingCap:
            def read(self):
                return False, None

        worker._run_capture_loop(FailingCap(), object())
        assert worker.error == "lost camera, reconnecting…"


class TestMain:
    def test_main_runs_app(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "request_camera_access", lambda: None)
        run_calls = []
        monkeypatch.setattr(app.MooditoApp, "run", lambda self: run_calls.append(True))
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        app.main()
        assert run_calls == [True]


