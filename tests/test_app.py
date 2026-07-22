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
    monkeypatch.setattr(app, "LICENSE_PATH", str(tmp_path / "license.json"))
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


class TestAppVersion:
    def test_uses_env_var_when_not_frozen(self, monkeypatch) -> None:
        monkeypatch.setattr(app.sys, "frozen", False, raising=False)
        monkeypatch.setenv("MOODITO_VERSION", "9.9.9")
        assert app.app_version() == "9.9.9"

    def test_defaults_to_dev_without_env(self, monkeypatch) -> None:
        monkeypatch.setattr(app.sys, "frozen", False, raising=False)
        monkeypatch.delenv("MOODITO_VERSION", raising=False)
        assert app.app_version() == "dev"


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
        assert calls[0][0] == app.MODEL_URL
        assert calls[0][1] != app.MODEL_PATH
        assert (data_dir / "model.task").exists()

    def test_failed_download_removes_partial_file(
        self, data_dir, monkeypatch
    ) -> None:
        def fail_download(_url, path):
            with open(path, "wb") as model_file:
                model_file.write(b"partial")
            raise OSError("connection dropped")

        monkeypatch.setattr(app.urllib.request, "urlretrieve", fail_download)

        with pytest.raises(OSError, match="connection dropped"):
            app.ensure_model()

        assert not (data_dir / "model.task").exists()
        assert list(data_dir.iterdir()) == []


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
        assert app.format_timestamp("2026-06-21T22:10:00") == "Jun 21, 2026 22:10"

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
        assert app.append_raw_samples([]) is True
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
        assert app.append_raw_samples([("t", "happy", "0.9")]) is False


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


class TestPrivacyAudio:
    def test_lock_screen_uses_native_login_service(self, monkeypatch) -> None:
        calls = []

        class FakeLockScreen:
            argtypes = None
            restype = object()

            def __call__(self):
                calls.append("lock")

        lock_screen = FakeLockScreen()
        framework = type(
            "FakeLoginFramework",
            (),
            {"SACLockScreenImmediate": lock_screen},
        )()
        loaded = []
        monkeypatch.setattr(
            app.ctypes,
            "CDLL",
            lambda path: loaded.append(path) or framework,
        )

        assert app.lock_screen_for_privacy() is True
        assert loaded == [
            "/System/Library/PrivateFrameworks/login.framework/Versions/A/login"
        ]
        assert lock_screen.argtypes == []
        assert lock_screen.restype is None
        assert calls == ["lock"]

    def test_lock_screen_handles_unavailable_login_service(self, monkeypatch) -> None:
        def unavailable(_path):
            raise OSError("framework unavailable")

        monkeypatch.setattr(app.ctypes, "CDLL", unavailable)
        assert app.lock_screen_for_privacy() is False

    def test_dim_screens_captures_levels_and_sets_all_brightness_to_zero(
        self, monkeypatch
    ) -> None:
        calls = []

        class FakeFunction:
            def __init__(self, callback):
                self.callback = callback
                self.argtypes = None
                self.restype = None

            def __call__(self, *args):
                return self.callback(*args)

        def get_online_displays(max_displays, display_ids, display_count):
            calls.append(("online", max_displays))
            display_count._obj.value = 2
            if display_ids is not None:
                display_ids[0] = 42
                display_ids[1] = 84
            return 0

        def get_brightness(display_id, level_pointer):
            calls.append(("get", display_id))
            level_pointer._obj.value = {42: 0.75, 84: 0.5}[display_id]
            return 0

        def set_brightness(display_id, level):
            calls.append(("set", display_id, level))
            return 0

        core_graphics = type(
            "FakeCoreGraphics",
            (),
            {"CGGetOnlineDisplayList": FakeFunction(get_online_displays)},
        )()
        display_services = type(
            "FakeDisplayServices",
            (),
            {
                "DisplayServicesGetBrightness": FakeFunction(get_brightness),
                "DisplayServicesSetBrightness": FakeFunction(set_brightness),
            },
        )()

        def load_framework(path):
            if path.endswith("CoreGraphics.framework/CoreGraphics"):
                return core_graphics
            if path.endswith("DisplayServices.framework/DisplayServices"):
                return display_services
            raise OSError(path)

        monkeypatch.setattr(app.ctypes, "CDLL", load_framework)

        assert app.dim_screens_for_privacy() == (
            app.ScreenBrightnessState(display_id=42, level=0.75),
            app.ScreenBrightnessState(display_id=84, level=0.5),
        )
        assert calls == [
            ("online", 0),
            ("online", 2),
            ("get", 42),
            ("set", 42, 0.0),
            ("get", 84),
            ("set", 84, 0.0),
        ]

    def test_dim_screens_skips_external_and_dims_mirrored_builtin(
        self, monkeypatch
    ) -> None:
        calls = []

        def get_brightness(display_id, level_pointer):
            calls.append(("get", display_id))
            if display_id == 2:
                return 1
            level_pointer._obj.value = 0.5
            return 0

        def set_brightness(display_id, level):
            calls.append(("set", display_id, level))
            return 0

        monkeypatch.setattr(
            app,
            "_screen_brightness_api",
            lambda: (object(), get_brightness, set_brightness),
        )
        monkeypatch.setattr(app, "_online_display_ids", lambda _function: (2, 1))

        assert app.dim_screens_for_privacy() == (
            app.ScreenBrightnessState(display_id=1, level=0.5),
        )
        assert calls == [("get", 2), ("get", 1), ("set", 1, 0.0)]

    def test_restore_screen_brightness_uses_captured_display_and_level(
        self, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            app,
            "_screen_brightness_api",
            lambda: (
                None,
                None,
                lambda display_id, level: calls.append((display_id, level)) or 0,
            ),
        )

        state = app.ScreenBrightnessState(display_id=42, level=0.75)
        assert app.restore_screen_brightness(state) is True
        assert calls == [(42, 0.75)]

    def test_capture_reads_only_selected_channels(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "_run_osascript",
            lambda *statements: (True, "42,false,67"),
        )
        state = app._capture_audio_state(microphone=True, speakers=False)
        assert state == app.AudioState(input_volume=67)

    def test_capture_rejects_unexpected_output(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "_run_osascript",
            lambda *statements: (True, "not audio settings"),
        )
        assert app._capture_audio_state(True, True) is None

    def test_mute_returns_snapshot_and_mutes_selected_channels(
        self, monkeypatch
    ) -> None:
        state = app.AudioState(input_volume=55, output_volume=40, output_muted=False)
        monkeypatch.setattr(app, "_capture_audio_state", lambda *args: state)
        calls = []
        monkeypatch.setattr(
            app,
            "_run_osascript",
            lambda *statements: calls.append(statements) or (True, ""),
        )
        assert app.mute_audio_for_privacy(True, True) == app.PrivacyMuteResult(
            state, applied=True
        )
        assert calls == [
            ("set volume input volume 0", "set volume with output muted")
        ]

    def test_restore_reinstates_volume_and_prior_mute_state(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            app,
            "_run_osascript",
            lambda *statements: calls.append(statements) or (True, ""),
        )
        state = app.AudioState(input_volume=61, output_volume=35, output_muted=True)
        assert app.restore_audio_state(state) is True
        assert calls == [
            (
                "set volume input volume 61",
                "set volume output volume 35",
                "set volume with output muted",
            )
        ]

    def test_failed_mute_keeps_snapshot_when_rollback_fails(
        self, monkeypatch
    ) -> None:
        state = app.AudioState(input_volume=50, output_volume=30, output_muted=False)
        monkeypatch.setattr(app, "_capture_audio_state", lambda *args: state)
        monkeypatch.setattr(app, "_run_osascript", lambda *statements: (False, ""))
        monkeypatch.setattr(app, "restore_audio_state", lambda value: False)
        assert app.mute_audio_for_privacy(True, True) == app.PrivacyMuteResult(
            state, applied=False
        )


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
    inst._stats_range_start = app.datetime(2026, 6, 21, 22, 0, 0)
    inst._stats_range_end = None
    inst._stats_live_24h = False
    inst._settings = {}
    inst._notifications = dict.fromkeys(app.NOTIFICATION_KEYS, False)
    inst._last_notified_emotion = None
    inst._privacy = _privacy_config()
    inst._privacy_trigger_since = dict.fromkeys(app.PRIVACY_TRIGGERS)
    inst._privacy_active_trigger = None
    inst._privacy_attempted = {
        trigger: set() for trigger in app.PRIVACY_TRIGGERS
    }
    inst._privacy_audio_states = {}
    inst._privacy_changed_channels = set()
    inst._privacy_brightness_states = ()
    inst._privacy_stepper_handler = None
    inst._license_active = True
    inst._range_stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
    inst._range_last_state = None
    inst._hourly_activity = [0.0] * 24
    inst._hourly_emotion = {k: [0.0] * 24 for k in app.STAT_KEYS}
    inst._stats_items = {k: rumps.MenuItem(k) for k in app.STAT_KEYS}
    inst._stats_header_item = rumps.MenuItem("header")
    inst._stats_total_item = rumps.MenuItem("total")
    inst._stats_since_item = rumps.MenuItem("since")
    inst._stats_range_item = rumps.MenuItem("range", callback=inst.set_stats_range)
    inst._stats_live_item = rumps.MenuItem("live")
    inst._stats_activity_header_item = rumps.MenuItem("activity")
    inst._stats_activity_item = rumps.MenuItem("spark")
    inst._stats_activity_axis_item = rumps.MenuItem("axis")
    inst._stats_heatmap_header_item = rumps.MenuItem("heatmap")
    inst._stats_heatmap_items = {k: rumps.MenuItem(f"h-{k}") for k in app.STAT_KEYS}
    inst._stats_heatmap_axis_item = rumps.MenuItem("heataxis")
    inst._stats_reset_item = rumps.MenuItem("reset")
    return inst


def _privacy_config(no_face=None, multi_face=None):
    defaults = {
        "microphone_seconds": 0,
        "speakers_seconds": 0,
        "screen_brightness_seconds": 0,
        "lock_screen_seconds": 0,
    }
    privacy = {
        "no_face": dict(defaults),
        "multi_face": dict(defaults),
    }
    privacy["no_face"].update(no_face or {})
    privacy["multi_face"].update(multi_face or {})
    return privacy


class TestStatsAccumulation:
    def test_insights_rows_follow_display_order(self, full_app) -> None:
        expected = [
            "neutral",
            "happy",
            "surprised",
            "angry",
            "sad",
            app.NO_FACE_LABEL,
            app.MULTI_FACE_LABEL,
            "paused",
            "error",
        ]
        assert app.STAT_KEYS == expected
        assert list(full_app._stats_heatmap_items) == expected
        assert list(full_app._stats_items) == expected
        assert app.STAT_EMOJI[app.MULTI_FACE_LABEL] == "👥"
        assert full_app._stats_items[app.MULTI_FACE_LABEL]._menuitem.image() is not None
        assert (
            full_app._stats_heatmap_items[app.MULTI_FACE_LABEL]._menuitem.image()
            is not None
        )

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
        monkeypatch.setattr(app, "append_raw_samples", lambda *a, **k: True)
        inst = _bare_app()
        ticks = int(10 / app.UI_REFRESH_INTERVAL) + 1
        for _ in range(ticks):
            inst._accumulate_stats("happy")
        assert saved  # at least one periodic flush occurred

    def test_update_stats_menu_sets_titles(self) -> None:
        inst = _bare_app()
        inst._range_stats["happy"] = {"seconds": 30.0, "count": 2}
        inst._update_stats_menu()
        assert "Since" in inst._stats_since_item.title
        assert "Now" in inst._stats_range_item.title
        assert inst._stats_reset_item.title == "Erase"


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
            app, "append_raw_samples", lambda rows: flushed.extend(rows) or True
        )
        inst = _bare_app()
        ticks = int(10 / app.UI_REFRESH_INTERVAL) + 1
        for _ in range(ticks):
            inst._accumulate_stats("happy", 0.5)
        assert flushed  # raw samples were flushed
        assert inst._raw_buffer == []  # buffer cleared after flush

    def test_periodic_flush_retains_buffer_on_write_failure(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        monkeypatch.setattr(app, "append_raw_samples", lambda _rows: False)
        inst = _bare_app()
        ticks = int(10 / app.UI_REFRESH_INTERVAL) + 1

        for _ in range(ticks):
            inst._accumulate_stats("happy", 0.5)

        assert len(inst._raw_buffer) == ticks


class TestStatsRange:
    def test_parse_datetime_input_accepts_common_formats(self) -> None:
        assert app.parse_datetime_input("2026-06-21 22:10") == app.datetime(
            2026, 6, 21, 22, 10
        )
        assert app.parse_datetime_input("2026-06-21T22:10:05") == app.datetime(
            2026, 6, 21, 22, 10, 5
        )
        assert app.parse_datetime_input("2026-06-21") == app.datetime(
            2026, 6, 21, 0, 0
        )

    def test_parse_datetime_input_rejects_garbage(self) -> None:
        assert app.parse_datetime_input("not a date") is None
        assert app.parse_datetime_input("") is None
        assert app.parse_datetime_input(None) is None

    def test_parse_datetime_range_accepts_now_end(self) -> None:
        start, end = app.parse_datetime_range("2026-06-21 22:00 to now")
        assert start == app.datetime(2026, 6, 21, 22, 0)
        assert end is None

    def test_parse_datetime_range_accepts_fixed_end(self) -> None:
        start, end = app.parse_datetime_range(
            "2026-06-21 22:00 to 2026-06-22 08:00"
        )
        assert start == app.datetime(2026, 6, 21, 22, 0)
        assert end == app.datetime(2026, 6, 22, 8, 0)

    def test_parse_datetime_range_accepts_begin_start(self) -> None:
        start, end = app.parse_datetime_range("begin to 2026-06-22 08:00")
        assert start is None  # 'begin' resolves to the Since datetime
        assert end == app.datetime(2026, 6, 22, 8, 0)

    def test_parse_datetime_range_accepts_begin_to_now(self) -> None:
        start, end = app.parse_datetime_range("begin to now")
        assert start is None
        assert end is None

    def test_parse_datetime_range_rejects_garbage(self) -> None:
        assert app.parse_datetime_range("garbage") is None
        assert app.parse_datetime_range("2026-06-21 to bad") is None
        assert app.parse_datetime_range("") is None

    def test_set_range_begin_resolves_to_since(
        self, data_dir, monkeypatch
    ) -> None:
        _patch_window(monkeypatch, clicked=1, text="begin to now")
        inst = _bare_app()
        inst._stats_started_at = "2026-06-21T22:00:00"
        inst.set_stats_range(None)
        assert inst._stats_range_start == app.datetime(2026, 6, 21, 22, 0)
        assert inst._stats_range_end is None

    def test_render_activity_sparkline_shapes_and_scaling(self) -> None:
        hourly = [0.0] * 24
        hourly[9] = 60.0  # busiest hour
        hourly[12] = 30.0
        line = app.render_activity_sparkline(hourly)
        assert len(line) == 24
        # The busiest hour is the full block.
        assert line[9] == "█"
        # A half-height hour is a mid-level block, not full and not blank.
        assert line[12] not in (" ", "█")
        # Empty hours render as blanks.
        assert line[0] == " "

    def test_render_activity_sparkline_all_zero_is_blank(self) -> None:
        assert app.render_activity_sparkline([0.0] * 24) == " " * 24

    def test_render_emotion_heatmap_shapes_and_scaling(self) -> None:
        heat = {k: [0.0] * 24 for k in app.STAT_KEYS}
        heat["happy"][9] = 60.0  # busiest cell -> solid
        heat["sad"][9] = 15.0  # quarter -> light shade
        rows = app.render_emotion_heatmap(heat, app.STAT_KEYS)
        assert len(rows) == len(app.STAT_KEYS)
        assert all(len(r) == 24 for r in rows)
        happy = rows[app.STAT_KEYS.index("happy")]
        sad = rows[app.STAT_KEYS.index("sad")]
        assert happy[9] == "█"  # solid
        assert sad[9] not in (" ", "█")  # a partial shade
        assert happy[0] == " "  # empty cell is blank

    def test_render_emotion_heatmap_all_zero_is_blank(self) -> None:
        heat = {k: [0.0] * 24 for k in app.STAT_KEYS}
        rows = app.render_emotion_heatmap(heat, app.STAT_KEYS)
        assert all(r == " " * 24 for r in rows)

    def test_set_default_range_is_last_24h_live(self) -> None:
        inst = _bare_app()
        inst._stats_started_at = "2020-01-01T00:00:00"
        inst._set_default_range()
        assert inst._stats_range_end is None  # live ("now")
        expected = app.datetime.now() - app.timedelta(hours=24)
        assert abs((inst._stats_range_start - expected).total_seconds()) < 5

    def test_set_default_range_clamps_to_tracking_start(self) -> None:
        inst = _bare_app()
        recent = (app.datetime.now() - app.timedelta(hours=1)).isoformat(
            timespec="seconds"
        )
        inst._stats_started_at = recent
        inst._set_default_range()
        # Tracking only began 1h ago, so the window cannot start 24h ago.
        assert inst._stats_range_start == app.datetime.fromisoformat(recent)

    def test_recompute_aggregates_only_in_range_rows(self, data_dir) -> None:
        (data_dir / "raw_data.csv").write_text(
            "timestamp,state,score\n"
            "2026-06-21T22:00:00,happy,0.9\n"
            "2026-06-21T22:00:01,happy,0.9\n"
            "2026-06-21T22:00:02,sad,0.5\n"
            "2026-06-20T10:00:00,happy,0.9\n"  # before range
            "2026-06-30T10:00:00,happy,0.9\n"  # after range
        )
        inst = _bare_app()
        inst._stats_range_start = app.datetime(2026, 6, 21, 0, 0)
        inst._stats_range_end = app.datetime(2026, 6, 22, 0, 0)
        inst._recompute_range_stats()
        assert inst._range_stats["happy"]["count"] == 1
        assert inst._range_stats["happy"]["seconds"] == pytest.approx(
            2 * app.UI_REFRESH_INTERVAL
        )
        assert inst._range_stats["sad"]["count"] == 1
        assert inst._range_stats["sad"]["seconds"] == pytest.approx(
            app.UI_REFRESH_INTERVAL
        )

    def test_recompute_includes_buffered_rows(self, data_dir) -> None:
        inst = _bare_app()
        inst._stats_range_start = app.datetime(2026, 6, 21, 0, 0)
        inst._stats_range_end = app.datetime(2026, 6, 22, 0, 0)
        inst._raw_buffer = [("2026-06-21T23:00:00.000", "happy", "0.9")]
        inst._recompute_range_stats()
        assert inst._range_stats["happy"]["count"] == 1

    def test_recompute_buckets_hourly_activity(self, data_dir) -> None:
        (data_dir / "raw_data.csv").write_text(
            "timestamp,state,score\n"
            "2026-06-21T09:00:00,happy,0.9\n"
            "2026-06-21T09:30:00,sad,0.5\n"  # same hour bucket as above
            "2026-06-21T18:00:00,paused,\n"  # paused still counts as usage
            "2026-06-20T18:00:00,happy,0.9\n"  # before range, ignored
        )
        inst = _bare_app()
        inst._stats_range_start = app.datetime(2026, 6, 21, 0, 0)
        inst._stats_range_end = app.datetime(2026, 6, 22, 0, 0)
        inst._recompute_range_stats()
        assert inst._hourly_activity[9] == pytest.approx(2 * app.UI_REFRESH_INTERVAL)
        assert inst._hourly_activity[18] == pytest.approx(app.UI_REFRESH_INTERVAL)
        assert inst._hourly_activity[0] == 0.0

    def test_recompute_buckets_emotion_heatmap(self, data_dir) -> None:
        (data_dir / "raw_data.csv").write_text(
            "timestamp,state,score\n"
            "2026-06-21T09:00:00,happy,0.9\n"
            "2026-06-21T09:30:00,happy,0.9\n"  # same hour bucket
            "2026-06-21T18:00:00,sad,0.5\n"
        )
        inst = _bare_app()
        inst._stats_range_start = app.datetime(2026, 6, 21, 0, 0)
        inst._stats_range_end = app.datetime(2026, 6, 22, 0, 0)
        inst._recompute_range_stats()
        assert inst._hourly_emotion["happy"][9] == pytest.approx(
            2 * app.UI_REFRESH_INTERVAL
        )
        assert inst._hourly_emotion["sad"][18] == pytest.approx(
            app.UI_REFRESH_INTERVAL
        )
        assert inst._hourly_emotion["happy"][0] == 0.0

    def test_live_accumulate_updates_range_stats(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        inst = _bare_app()
        inst._stats_range_start = app.datetime(2020, 1, 1)
        inst._stats_range_end = None  # live window
        inst._accumulate_stats("happy", 0.9)
        assert inst._range_stats["happy"]["seconds"] == pytest.approx(
            app.UI_REFRESH_INTERVAL
        )
        assert inst._range_stats["happy"]["count"] == 1

    def test_fixed_end_range_ignores_live_samples(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        inst = _bare_app()
        inst._stats_range_start = app.datetime(2020, 1, 1)
        inst._stats_range_end = app.datetime(2020, 1, 2)  # fixed, in the past
        inst._accumulate_stats("happy", 0.9)
        assert inst._range_stats["happy"]["seconds"] == pytest.approx(0.0)

    def test_set_range_clamps_start_before_tracking_start(
        self, data_dir, monkeypatch
    ) -> None:
        _patch_window(monkeypatch, clicked=1, text="2020-01-01 00:00 to now")
        inst = _bare_app()
        inst._stats_started_at = "2026-06-21T22:00:00"
        inst.set_stats_range(None)
        assert inst._stats_range_start == app.datetime(2026, 6, 21, 22, 0)
        assert inst._stats_range_end is None

    def test_set_range_clamps_end_to_now(self, data_dir, monkeypatch) -> None:
        _patch_window(
            monkeypatch, clicked=1, text="2026-06-21 22:00 to 2099-01-01 00:00"
        )
        inst = _bare_app()
        inst.set_stats_range(None)
        assert inst._stats_range_end is not None
        assert inst._stats_range_end <= app.datetime.now()

    def test_set_range_rejects_start_equal_end(
        self, data_dir, monkeypatch
    ) -> None:
        _patch_window(
            monkeypatch, clicked=1, text="2026-06-21 22:00 to 2026-06-21 22:00"
        )
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        inst = _bare_app()
        before_start = inst._stats_range_start
        before_end = inst._stats_range_end
        inst.set_stats_range(None)
        assert alerts  # user was warned
        assert inst._stats_range_start == before_start  # unchanged
        assert inst._stats_range_end == before_end

    def test_set_range_rejects_start_after_end(
        self, data_dir, monkeypatch
    ) -> None:
        _patch_window(
            monkeypatch, clicked=1, text="2026-06-21 23:00 to 2026-06-21 22:00"
        )
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        inst = _bare_app()
        before_start = inst._stats_range_start
        inst.set_stats_range(None)
        assert alerts  # user was warned
        assert inst._stats_range_start == before_start  # unchanged

    def test_set_range_rejects_start_after_now_with_live_end(
        self, data_dir, monkeypatch
    ) -> None:
        # A live ("now") end must still reject a start that is after now.
        _patch_window(monkeypatch, clicked=1, text="2099-01-01 00:00 to now")
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        inst = _bare_app()
        before_start = inst._stats_range_start
        inst.set_stats_range(None)
        assert alerts  # user was warned
        assert inst._stats_range_start == before_start  # unchanged

    def test_set_range_invalid_input_shows_alert(
        self, data_dir, monkeypatch
    ) -> None:
        _patch_window(monkeypatch, clicked=1, text="garbage")
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        inst = _bare_app()
        before = inst._stats_range_start
        inst.set_stats_range(None)
        assert alerts  # user was warned
        assert inst._stats_range_start == before  # unchanged

    def test_set_range_cancel_keeps_range(self, data_dir, monkeypatch) -> None:
        _patch_window(
            monkeypatch, clicked=0, text="2026-06-21 23:00 to now"
        )
        inst = _bare_app()
        before = inst._stats_range_start
        inst.set_stats_range(None)
        assert inst._stats_range_start == before

    def test_toggle_live_24h_on_pins_window_and_locks_range(
        self, data_dir, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda *a, **k: None)
        inst = _bare_app()
        inst._stats_started_at = "2020-01-01T00:00:00"
        inst._stats_range_end = app.datetime(2021, 1, 1)
        assert inst._stats_live_24h is False
        inst.toggle_live_24h(inst._stats_live_item)
        assert inst._stats_live_24h is True
        assert inst._stats_live_item.state == 1
        # Window pinned to the live last 24 hours.
        assert inst._stats_range_end is None
        expected = app.datetime.now() - app.timedelta(hours=24)
        assert abs((inst._stats_range_start - expected).total_seconds()) < 5
        # Manual Range control is locked (no callback).
        assert inst._stats_range_item.callback is None

    def test_toggle_live_24h_off_unlocks_range(
        self, data_dir, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda *a, **k: None)
        inst = _bare_app()
        inst._stats_live_24h = True
        inst.toggle_live_24h(inst._stats_live_item)
        assert inst._stats_live_24h is False
        assert inst._stats_live_item.state == 0
        # Manual Range control is editable again.
        assert inst._stats_range_item.callback is not None

    def test_set_range_ignored_while_live(self, data_dir, monkeypatch) -> None:
        _patch_window(monkeypatch, clicked=1, text="2026-06-21 22:00 to now")
        inst = _bare_app()
        inst._stats_live_24h = True
        before = inst._stats_range_start
        inst.set_stats_range(None)
        assert inst._stats_range_start == before  # unchanged while locked

    def test_toggle_live_24h_off_blocked_without_license(
        self, data_dir, monkeypatch
    ) -> None:
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        inst = _bare_app()
        inst._stats_live_24h = True
        inst._license_active = False
        inst.toggle_live_24h(inst._stats_live_item)
        assert alerts  # user was told a license is required
        assert inst._stats_live_24h is True  # stays on
        assert inst._stats_live_item.state == 1  # stays checked

    def test_toggle_live_24h_on_allowed_without_license(
        self, data_dir, monkeypatch
    ) -> None:
        # Re-enabling the live window never requires a license.
        inst = _bare_app()
        inst._stats_live_24h = False
        inst._license_active = False
        inst.toggle_live_24h(inst._stats_live_item)
        assert inst._stats_live_24h is True



class TestExportCsv:
    def test_exports_rows_in_selected_range(
        self, data_dir, monkeypatch
    ) -> None:
        # Pretend HOME is the temp dir and create a Downloads folder.
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        (data_dir / "raw_data.csv").write_text(
            "timestamp,state,score\n2026-06-21T22:30:00,happy,0.9\n"
        )
        opened = []
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: opened.append(a))
        inst = _bare_app()
        inst.export_csv(None)
        exported = list((data_dir / "Downloads").glob("moodito-raw-*.csv"))
        assert len(exported) == 1
        assert "happy" in exported[0].read_text()
        assert opened and opened[0][0][0] == "open"

    def test_excludes_rows_outside_selected_range(
        self, data_dir, monkeypatch
    ) -> None:
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        (data_dir / "raw_data.csv").write_text(
            "timestamp,state,score\n"
            "2026-06-21T22:30:00,happy,0.9\n"  # in range
            "2020-01-01T00:00:00,sad,0.5\n"  # before range
            "2099-01-01T00:00:00,angry,0.5\n"  # after range
        )
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: None)
        inst = _bare_app()
        inst._stats_range_start = app.datetime(2026, 6, 21, 0, 0)
        inst._stats_range_end = app.datetime(2026, 6, 22, 0, 0)
        inst.export_csv(None)
        exported = list((data_dir / "Downloads").glob("moodito-raw-*.csv"))
        text = exported[0].read_text()
        assert "happy" in text
        assert "sad" not in text
        assert "angry" not in text

    def test_flushes_buffer_before_export(self, data_dir, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: None)
        inst = _bare_app()
        inst._raw_buffer = [("2026-06-21T22:30:00.000", "happy", "0.9")]
        inst.export_csv(None)
        assert inst._raw_buffer == []
        exported = list((data_dir / "Downloads").glob("moodito-raw-*.csv"))
        assert "happy" in exported[0].read_text()

    def test_failed_buffer_flush_aborts_export(
        self, data_dir, monkeypatch
    ) -> None:
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        monkeypatch.setattr(app, "append_raw_samples", lambda _rows: False)
        inst = _bare_app()
        inst._raw_buffer = [("2026-06-21T22:30:00.000", "happy", "0.9")]
        notifications = []
        monkeypatch.setattr(
            inst,
            "_deliver_notification",
            lambda *args: notifications.append(args),
        )

        inst.export_csv(None)

        assert inst._raw_buffer == [
            ("2026-06-21T22:30:00.000", "happy", "0.9")
        ]
        assert list((data_dir / "Downloads").iterdir()) == []
        assert notifications[0][0] == "csv_export_failed"

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
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: 1)
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
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: 1)
        inst = _bare_app()
        inst.reset_stats(None)  # no raw file present; must not raise

    def test_cancel_keeps_data(self, data_dir, monkeypatch) -> None:
        # Declining the confirmation must leave everything untouched.
        saved = []
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: saved.append(a))
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: 0)
        (data_dir / "raw_data.csv").write_text("timestamp,state,score\nt,happy,0.9\n")
        inst = _bare_app()
        inst._stats["happy"] = {"seconds": 30.0, "count": 5}
        inst._raw_buffer = [("t", "happy", "0.9")]
        inst.reset_stats(None)
        assert inst._stats["happy"] == {"seconds": 30.0, "count": 5}
        assert inst._raw_buffer == [("t", "happy", "0.9")]
        assert (data_dir / "raw_data.csv").exists()
        assert saved == []  # nothing persisted



class TestQuitApp:
    def test_stops_worker_and_flushes_raw(self, monkeypatch) -> None:
        saved = []
        flushed = []
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: saved.append(a))
        monkeypatch.setattr(
            app, "append_raw_samples", lambda rows: flushed.extend(rows) or True
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

    def test_write_failure_keeps_app_running_and_buffered(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        monkeypatch.setattr(app, "append_raw_samples", lambda _rows: False)
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        quit_calls = []
        monkeypatch.setattr(
            app.rumps, "quit_application", lambda: quit_calls.append(True)
        )
        inst = _bare_app()
        stopped = []
        inst._worker = type("W", (), {"stop": lambda self: stopped.append(True)})()
        inst._raw_buffer = [("t", "happy", "0.9")]

        inst.quit_app(None)

        assert inst._raw_buffer == [("t", "happy", "0.9")]
        assert stopped == []
        assert quit_calls == []
        assert alerts

    def test_privacy_restore_failure_keeps_app_running_by_default(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        append_calls = []
        monkeypatch.setattr(
            app,
            "append_raw_samples",
            lambda rows: append_calls.append(rows) or True,
        )
        alerts = []

        def stay_open(*args, **kwargs):
            alerts.append((args, kwargs))
            return 1

        monkeypatch.setattr(app.rumps, "alert", stay_open)
        quit_calls = []
        monkeypatch.setattr(
            app.rumps, "quit_application", lambda: quit_calls.append(True)
        )
        inst = _bare_app()
        monkeypatch.setattr(inst, "_reset_privacy", lambda: False)
        stopped = []
        inst._worker = type("W", (), {"stop": lambda self: stopped.append(True)})()

        inst.quit_app(None)

        assert append_calls == []
        assert stopped == []
        assert quit_calls == []
        assert alerts[0][1] == {"ok": "Stay Open", "cancel": "Quit Anyway"}

    def test_privacy_restore_failure_can_be_overridden(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_stats", lambda *a, **k: None)
        monkeypatch.setattr(app, "append_raw_samples", lambda _rows: True)
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: 0)
        quit_calls = []
        monkeypatch.setattr(
            app.rumps, "quit_application", lambda: quit_calls.append(True)
        )
        inst = _bare_app()
        monkeypatch.setattr(inst, "_reset_privacy", lambda: False)
        stopped = []
        inst._worker = type("W", (), {"stop": lambda self: stopped.append(True)})()

        inst.quit_app(None)

        assert stopped == [True]
        assert quit_calls == [True]


class TestLicensePersistence:
    def test_load_missing_returns_empty(self, data_dir) -> None:
        assert app.load_license() == {}

    def test_save_then_load_roundtrip(self, data_dir) -> None:
        data = {"license_key": "abc", "instance_id": "xyz", "instance_name": "Moodito"}
        app.save_license(data)
        assert app.load_license() == data

    def test_non_dict_json_returns_empty(self, data_dir) -> None:
        (data_dir / "license.json").write_text("[1, 2, 3]")
        assert app.load_license() == {}

    def test_corrupt_json_returns_empty(self, data_dir) -> None:
        (data_dir / "license.json").write_text("{broken")
        assert app.load_license() == {}

    def test_clear_removes_file(self, data_dir) -> None:
        app.save_license({"license_key": "abc", "instance_id": "xyz"})
        app.clear_license()
        assert not (data_dir / "license.json").exists()

    def test_clear_missing_file_is_tolerated(self, data_dir) -> None:
        app.clear_license()  # must not raise

    def test_save_swallows_os_error(self, data_dir, monkeypatch) -> None:
        monkeypatch.setattr(
            app.os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(OSError())
        )
        app.save_license({"license_key": "abc"})  # best-effort, no raise

    def test_is_license_active(self) -> None:
        assert app.is_license_active({"license_key": "k", "instance_id": "i"}) is True
        assert app.is_license_active({"license_key": "k"}) is False
        assert app.is_license_active({"instance_id": "i"}) is False
        assert app.is_license_active({}) is False

    def test_instance_name_includes_hostname(self, monkeypatch) -> None:
        monkeypatch.setattr(app.socket, "gethostname", lambda: "MyMac")
        assert app.license_instance_name() == "Moodito on MyMac"

    def test_instance_name_tolerates_failure(self, monkeypatch) -> None:
        def boom():
            raise OSError("no host")

        monkeypatch.setattr(app.socket, "gethostname", boom)
        assert app.license_instance_name() == "Moodito on Mac"


class TestLicenseApi:
    def test_activate_success(self, monkeypatch) -> None:
        captured = {}

        def fake_request(action, params):
            captured["action"] = action
            captured["params"] = params
            return {"activated": True, "instance": {"id": "iid"}}

        monkeypatch.setattr(app, "_license_api_request", fake_request)
        ok, message, instance_id = app.activate_license("key-1", "Moodito on Mac")
        assert ok is True
        assert message == "activated"
        assert instance_id == "iid"
        assert captured["action"] == "activate"
        assert captured["params"] == {
            "license_key": "key-1",
            "instance_name": "Moodito on Mac",
        }

    def test_activate_rejected_returns_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "_license_api_request",
            lambda *a, **k: {"activated": False, "error": "limit reached"},
        )
        ok, message, instance_id = app.activate_license("key-1", "Moodito")
        assert ok is False
        assert message == "limit reached"
        assert instance_id == ""

    def test_activate_missing_instance_id_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "_license_api_request",
            lambda *a, **k: {"activated": True, "instance": {}},
        )
        ok, message, instance_id = app.activate_license("key-1", "Moodito")
        assert ok is False
        assert "instance id" in message
        assert instance_id == ""

    def test_activate_network_error(self, monkeypatch) -> None:
        def boom(*_a, **_k):
            raise OSError("offline")

        monkeypatch.setattr(app, "_license_api_request", boom)
        ok, message, instance_id = app.activate_license("key-1", "Moodito")
        assert ok is False
        assert "license server" in message
        assert instance_id == ""

    def test_validate_valid(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app, "_license_api_request", lambda *a, **k: {"valid": True}
        )
        assert app.validate_license("key-1", "iid") == app.LICENSE_VALID

    def test_validate_invalid(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "_license_api_request",
            lambda *a, **k: {"valid": False, "error": "expired"},
        )
        assert app.validate_license("key-1", "iid") == app.LICENSE_INVALID

    def test_validate_network_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "_license_api_request",
            lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
        )
        assert app.validate_license("key-1", "iid") == app.LICENSE_UNREACHABLE

    @pytest.mark.parametrize("status", [429, 503])
    def test_validate_transient_http_error_is_unreachable(
        self, monkeypatch, status
    ) -> None:
        import io

        def fake_urlopen(request, timeout):
            raise app.urllib.error.HTTPError(
                request.full_url,
                status,
                "Temporarily unavailable",
                {},
                io.BytesIO(b'{"error": "try again"}'),
            )

        monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)
        assert app.validate_license("key-1", "iid") == app.LICENSE_UNREACHABLE

    def test_deactivate_success(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app, "_license_api_request", lambda *a, **k: {"deactivated": True}
        )
        ok, message = app.deactivate_license("key-1", "iid")
        assert ok is True
        assert message == "deactivated"

    def test_deactivate_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app,
            "_license_api_request",
            lambda *a, **k: {"deactivated": False, "error": "not found"},
        )
        ok, message = app.deactivate_license("key-1", "iid")
        assert ok is False
        assert message == "not found"

    def test_api_request_posts_form_encoded(self, monkeypatch) -> None:
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"activated": true}'

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["data"] = request.data
            captured["method"] = request.get_method()
            captured["accept"] = request.get_header("Accept")
            return FakeResponse()

        monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)
        result = app._license_api_request("activate", {"license_key": "k"})
        assert result == {"activated": True}
        assert captured["url"] == f"{app.LICENSE_API_BASE}/activate"
        assert captured["data"] == b"license_key=k"
        assert captured["method"] == "POST"
        assert captured["accept"] == "application/json"

    def test_api_request_parses_http_error_body(self, monkeypatch) -> None:
        import io

        def fake_urlopen(request, timeout):
            raise app.urllib.error.HTTPError(
                request.full_url, 422, "Unprocessable", {},
                io.BytesIO(b'{"error": "invalid key"}'),
            )

        monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)
        result = app._license_api_request("activate", {"license_key": "k"})
        assert result == {"error": "invalid key"}


def _bare_license_app(active: bool = False):
    """Build a MooditoApp with just the attributes the license logic needs."""
    import rumps

    inst = object.__new__(app.MooditoApp)
    inst._license = (
        {"license_key": "key-1", "instance_id": "iid", "instance_name": "Moodito"}
        if active
        else {}
    )
    inst._license_active = active
    inst._license_lock = app.threading.Lock()
    inst._license_busy = app.threading.Event()
    inst._license_alert = None
    inst._license_notification_event = None
    inst._license_dirty = app.threading.Event()
    inst._notifications = dict.fromkeys(app.NOTIFICATION_KEYS, False)
    inst._license_status_item = rumps.MenuItem("status")
    inst._license_key_item = rumps.MenuItem("key")
    inst._license_device_item = rumps.MenuItem("device")
    inst._license_activate_item = rumps.MenuItem("activate")
    inst._license_deactivate_item = rumps.MenuItem("deactivate")
    inst._license_buy_item = rumps.MenuItem("buy")
    inst._bmc_menu = rumps.MenuItem("bmc")
    return inst


class _FakeWindowResponse:
    def __init__(self, clicked, text):
        self.clicked = clicked
        self.text = text


def _patch_window(monkeypatch, clicked, text):
    """Make rumps.Window(...).run() return a canned response."""

    class FakeWindow:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return _FakeWindowResponse(clicked, text)

    monkeypatch.setattr(app.rumps, "Window", FakeWindow)


def _patch_sync_threads(monkeypatch):
    """Run threads spawned by app code synchronously inside .start()."""

    class SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(app.threading, "Thread", SyncThread)


class TestLicenseMenu:
    def test_visibility_when_active(self) -> None:
        inst = _bare_license_app(active=True)
        inst._apply_license_visibility()
        assert inst._license_activate_item.hidden is True
        assert inst._license_deactivate_item.hidden is False
        assert inst._bmc_menu.hidden is True
        assert "Licensed" in inst._license_status_item.title
        # Buy License is hidden once licensed.
        assert inst._license_buy_item.hidden is True
        # License details are shown and populated while licensed.
        assert inst._license_key_item.hidden is False
        assert inst._license_device_item.hidden is False
        assert "key-1"[-4:] in inst._license_key_item.title
        assert "Moodito" in inst._license_device_item.title

    def test_visibility_when_inactive(self) -> None:
        inst = _bare_license_app(active=False)
        inst._apply_license_visibility()
        assert inst._license_activate_item.hidden is False
        assert inst._license_deactivate_item.hidden is True
        assert inst._bmc_menu.hidden is False
        assert "Not licensed" in inst._license_status_item.title
        # License details are hidden while unlicensed.
        assert inst._license_key_item.hidden is True
        assert inst._license_device_item.hidden is True
        # Buy License is shown while unlicensed.
        assert inst._license_buy_item.hidden is False

    def test_buy_and_restore_open_urls(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: calls.append(a[0]))
        inst = _bare_license_app()
        inst.buy_license(None)
        inst.restore_license(None)
        assert calls[0] == ["open", app.LICENSE_BUY_URL]
        assert calls[1] == ["open", app.LICENSE_RESTORE_URL]

    def test_activate_dialog_success(self, data_dir, monkeypatch) -> None:
        _patch_window(monkeypatch, clicked=1, text="  my-key  ")
        _patch_sync_threads(monkeypatch)
        monkeypatch.setattr(app, "license_instance_name", lambda: "Moodito on Mac")
        monkeypatch.setattr(
            app, "activate_license", lambda key, name: (True, "activated", "iid")
        )
        inst = _bare_license_app(active=False)
        inst.activate_license_dialog(None)
        assert inst._license_active is True
        assert inst._license["license_key"] == "my-key"
        assert inst._license["instance_id"] == "iid"
        assert app.load_license()["license_key"] == "my-key"
        assert inst._license_busy.is_set() is False
        assert inst._license_dirty.is_set() is True
        assert "Thank you" in inst._license_alert

    def test_activate_dialog_cancelled(self, data_dir, monkeypatch) -> None:
        _patch_window(monkeypatch, clicked=0, text="my-key")
        _patch_sync_threads(monkeypatch)
        called = []
        monkeypatch.setattr(
            app,
            "activate_license",
            lambda *a, **k: called.append(a) or (True, "", "iid"),
        )
        inst = _bare_license_app(active=False)
        inst.activate_license_dialog(None)
        assert called == []  # API not hit on cancel
        assert inst._license_active is False
        assert inst._license_busy.is_set() is False

    def test_activate_dialog_empty_key_ignored(self, data_dir, monkeypatch) -> None:
        _patch_window(monkeypatch, clicked=1, text="   ")
        _patch_sync_threads(monkeypatch)
        called = []
        monkeypatch.setattr(
            app,
            "activate_license",
            lambda *a, **k: called.append(a) or (True, "", "iid"),
        )
        inst = _bare_license_app(active=False)
        inst.activate_license_dialog(None)
        assert called == []
        assert inst._license_active is False

    def test_activate_dialog_busy_is_ignored(self, monkeypatch) -> None:
        windows = []

        class FakeWindow:
            def __init__(self, *a, **k):
                windows.append(True)

            def run(self):
                return _FakeWindowResponse(1, "key")

        monkeypatch.setattr(app.rumps, "Window", FakeWindow)
        inst = _bare_license_app(active=False)
        inst._license_busy.set()
        inst.activate_license_dialog(None)
        assert windows == []  # dialog never opened while busy

    def test_activate_worker_failure_sets_alert(self, data_dir, monkeypatch) -> None:
        monkeypatch.setattr(app, "license_instance_name", lambda: "Moodito")
        monkeypatch.setattr(
            app, "activate_license", lambda *a, **k: (False, "limit reached", "")
        )
        inst = _bare_license_app(active=False)
        inst._license_busy.set()
        inst._activate_worker("bad-key")
        assert inst._license_active is False
        assert "limit reached" in inst._license_alert
        assert inst._license_busy.is_set() is False
        assert inst._license_dirty.is_set() is True

    def test_deactivate_success(self, data_dir, monkeypatch) -> None:
        _patch_sync_threads(monkeypatch)
        app.save_license({"license_key": "key-1", "instance_id": "iid"})
        monkeypatch.setattr(app, "deactivate_license", lambda *a, **k: (True, "ok"))
        inst = _bare_license_app(active=True)
        inst.deactivate_license_action(None)
        assert inst._license_active is False
        assert inst._license == {}
        assert not (data_dir / "license.json").exists()
        assert "deactivated" in inst._license_alert
        assert inst._license_dirty.is_set() is True

    def test_deactivate_failure_keeps_license(self, data_dir, monkeypatch) -> None:
        _patch_sync_threads(monkeypatch)
        monkeypatch.setattr(
            app, "deactivate_license", lambda *a, **k: (False, "not found")
        )
        inst = _bare_license_app(active=True)
        inst.deactivate_license_action(None)
        assert inst._license_active is True  # unchanged on failure
        assert "not found" in inst._license_alert

    def test_consume_license_updates_applies_and_alerts(self, monkeypatch) -> None:
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        inst = _bare_license_app(active=True)
        inst._license_alert = "License activated. Thank you! 🎉"
        inst._license_dirty.set()
        inst._consume_license_updates()
        assert inst._license_dirty.is_set() is False
        assert inst._license_alert is None
        assert alerts and "Thank you" in alerts[0][1]
        # Visibility was applied from the (active) state.
        assert inst._bmc_menu.hidden is True

    def test_consume_license_updates_noop_when_clean(self, monkeypatch) -> None:
        alerts = []
        monkeypatch.setattr(app.rumps, "alert", lambda *a, **k: alerts.append(a))
        inst = _bare_license_app(active=False)
        inst._consume_license_updates()
        assert alerts == []

    def test_periodic_check_rechecks_when_active(self, monkeypatch) -> None:
        _patch_sync_threads(monkeypatch)
        monkeypatch.setattr(app, "clear_license", lambda: None)
        monkeypatch.setattr(
            app, "validate_license", lambda *a, **k: app.LICENSE_INVALID
        )
        inst = _bare_license_app(active=True)
        inst._periodic_license_check(None)
        assert inst._license_active is False
        assert inst._license_dirty.is_set() is True

    def test_periodic_check_skips_when_inactive(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(
            app, "validate_license", lambda *a, **k: called.append(a) or "valid"
        )
        inst = _bare_license_app(active=False)
        inst._periodic_license_check(None)
        assert called == []

    def test_recheck_valid_keeps_license(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app, "validate_license", lambda *a, **k: app.LICENSE_VALID
        )
        inst = _bare_license_app(active=True)
        inst._recheck_license()
        assert inst._license_active is True
        assert inst._license_dirty.is_set() is False

    def test_recheck_invalid_clears_license(self, data_dir, monkeypatch) -> None:
        app.save_license({"license_key": "key-1", "instance_id": "iid"})
        monkeypatch.setattr(
            app, "validate_license", lambda *a, **k: app.LICENSE_INVALID
        )
        inst = _bare_license_app(active=True)
        inst._recheck_license()
        assert inst._license_active is False
        assert inst._license == {}
        assert inst._license_dirty.is_set() is True
        assert not (data_dir / "license.json").exists()

    def test_recheck_unreachable_keeps_license(self, monkeypatch) -> None:
        monkeypatch.setattr(
            app, "validate_license", lambda *a, **k: app.LICENSE_UNREACHABLE
        )
        inst = _bare_license_app(active=True)
        inst._recheck_license()
        assert inst._license_active is True  # transient error → keep
        assert inst._license_dirty.is_set() is False



@pytest.fixture
def full_app(data_dir, monkeypatch):
    """Construct a full MooditoApp without starting the camera thread."""
    monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
    monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
    monkeypatch.setattr(app.rumps, "notification", lambda *args, **kwargs: None)
    return app.MooditoApp()


class TestMooditoAppInit:
    def test_builds_menu_and_records_start_time(self, full_app) -> None:
        assert full_app._stats_started_at is not None
        assert full_app._raw_buffer == []
        assert "Download (csv)" in full_app._stats_export_item.title
        assert set(full_app._stats_items) == set(app.STAT_KEYS)

    def test_version_footer_is_clickable(self, full_app) -> None:
        assert full_app._version_item.title == f"Moodito {app.app_version()}"
        assert full_app._version_item.callback is not None

    def test_about_precedes_final_quit_command(self, full_app) -> None:
        menu_titles = list(full_app.menu.keys())
        assert menu_titles[-2:] == [
            full_app._version_item.title,
            full_app._quit_item.title,
        ]

    def test_about_accessory_contains_supplied_description(self, full_app) -> None:
        _view, text_view = full_app._build_about_accessory()
        text = str(text_view.string())
        assert text == app.ABOUT_DESCRIPTION
        assert "Your mood, live in the menu bar." in text
        assert "100% on-device" in text
        assert "wellbeing report" in text

    def test_about_action_presents_version_author_and_description(
        self, full_app, monkeypatch
    ) -> None:
        import AppKit

        captured = {"buttons": [], "modal": False}

        class FakeAlert:
            @classmethod
            def alloc(cls):
                return cls()

            def init(self):
                return self

            def setMessageText_(self, value) -> None:
                captured["title"] = value

            def setInformativeText_(self, value) -> None:
                captured["info"] = value

            def addButtonWithTitle_(self, value) -> None:
                captured["buttons"].append(value)

            def setIcon_(self, _value) -> None:
                captured["icon"] = True

            def setAccessoryView_(self, value) -> None:
                captured["accessory"] = value

            def runModal(self) -> None:
                captured["modal"] = True

        class FakeImage:
            @classmethod
            def alloc(cls):
                return cls()

            def initByReferencingFile_(self, _path):
                return self

            def isValid(self) -> bool:
                return True

        fake_ns_app = type(
            "FakeNSApp",
            (),
            {"activateIgnoringOtherApps_": lambda self, _active: None},
        )()
        accessory = object()
        monkeypatch.setattr(AppKit, "NSAlert", FakeAlert)
        monkeypatch.setattr(AppKit, "NSImage", FakeImage)
        monkeypatch.setattr(AppKit, "NSApp", fake_ns_app)
        monkeypatch.setattr(
            full_app, "_build_about_accessory", lambda: (accessory, object())
        )

        full_app.open_about_window(None)

        assert captured["title"] == "Moodito"
        assert f"Version {app.app_version()}" in captured["info"]
        assert f"Author: {app.APP_AUTHOR}" in captured["info"]
        assert captured["buttons"] == ["Close"]
        assert captured["accessory"] is accessory
        assert captured["modal"] is True

    def test_restores_display_settings(self, data_dir, monkeypatch) -> None:
        app.save_settings({"show_emojis": False, "show_labels": True})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._show_emojis is False
        assert inst._show_labels is True
        assert inst._emojis_item.state == 0
        assert inst._labels_item.state == 1

    def test_legacy_icon_only_maps_to_both_off(self, data_dir, monkeypatch) -> None:
        app.save_settings({"icon_only": True})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._show_emojis is False
        assert inst._show_labels is False


class TestRefresh:
    def test_normal_updates_detected_and_buffers_raw(self, full_app) -> None:
        full_app._worker._set_result(EmotionResult("happy", 0.8))
        full_app.refresh(None)
        assert "happy" in full_app._detected_item.title
        assert full_app._raw_buffer[-1][1] == "happy"

    def test_multiple_faces_update_menu_detected_row_and_insights_state(
        self, full_app
    ) -> None:
        transitions = []
        full_app._notify_emotion_transition = (
            lambda label, face_count=1: transitions.append((label, face_count))
        )
        full_app._worker._set_result(
            EmotionResult(app.MULTI_FACE_LABEL, 1.0, face_count=3)
        )

        full_app.refresh(None)

        assert full_app.title == "👥 3 faces"
        assert full_app._detected_item.title == "Detected: 3 faces"
        assert full_app._raw_buffer[-1][1] == app.MULTI_FACE_LABEL
        assert full_app._stats[app.MULTI_FACE_LABEL]["count"] == 1
        assert transitions == [(app.MULTI_FACE_LABEL, 3)]

    def test_paused_records_paused_state(self, full_app) -> None:
        full_app._paused = True
        full_app.refresh(None)
        assert full_app._raw_buffer[-1][1] == "paused"

    def test_error_updates_detected_title(self, full_app) -> None:
        full_app._worker._set_error("camera gone")
        full_app.refresh(None)
        assert "error" in full_app._detected_item.title
        assert full_app._raw_buffer[-1][1] == "error"

    def test_routes_detected_state_to_privacy(self, full_app) -> None:
        observations = []
        full_app._update_privacy = lambda state: observations.append(state)
        full_app._worker._set_result(EmotionResult(app.NO_FACE_LABEL, 0.0))
        full_app.refresh(None)
        full_app._worker._set_result(
            EmotionResult(app.MULTI_FACE_LABEL, 1.0, face_count=3)
        )
        full_app.refresh(None)
        full_app._worker._set_result(EmotionResult("happy", 0.8))
        full_app.refresh(None)
        assert observations == [app.NO_FACE_LABEL, app.MULTI_FACE_LABEL, "happy"]

    def test_routes_face_presence_to_break_timer(self, full_app) -> None:
        observations = []
        full_app._update_break_timer = (
            lambda face_present: observations.append(face_present)
        )
        full_app._worker._set_result(EmotionResult(app.NO_FACE_LABEL, 0.0))
        full_app.refresh(None)
        full_app._worker._set_result(EmotionResult("happy", 0.8))
        full_app.refresh(None)
        assert observations == [False, True]


class TestRenderStatus:
    def test_emojis_and_labels_show_emoji_and_text(self, full_app) -> None:
        full_app._show_emojis = True
        full_app._show_labels = True
        full_app._last_render = None
        full_app._render_emotion(EmotionResult("happy", 0.8))
        assert full_app.title == "😀 happy"
        assert full_app.icon is None

    def test_multiple_faces_show_icon_and_dynamic_count(self, full_app) -> None:
        full_app._show_emojis = True
        full_app._show_labels = True
        full_app._last_render = None

        full_app._render_emotion(
            EmotionResult(app.MULTI_FACE_LABEL, 1.0, face_count=3)
        )

        assert full_app.title == "👥 3 faces"
        assert full_app.icon is None

    def test_emojis_only_shows_emoji(self, full_app) -> None:
        full_app._show_emojis = True
        full_app._show_labels = False
        full_app._last_render = None
        full_app._render_emotion(EmotionResult("happy", 0.8))
        assert full_app.title == "😀"

    def test_no_emojis_with_labels_shows_icon_and_label(self, full_app) -> None:
        full_app._show_emojis = False
        full_app._show_labels = True
        full_app._last_render = None
        full_app._render_emotion(EmotionResult("happy", 0.8))
        assert full_app.icon is not None
        assert full_app.title == "happy"

    def test_no_emojis_no_labels_shows_icon_only(self, full_app) -> None:
        full_app._show_emojis = False
        full_app._show_labels = False
        full_app._last_render = None
        full_app._render_emotion(EmotionResult("happy", 0.8))
        assert full_app.icon is not None
        assert full_app.title is None

    def test_active_break_timer_shows_clock_in_top_menu(self, full_app) -> None:
        full_app._break_timer["duration_seconds"] = 60
        full_app._last_render = None
        full_app._render_emotion(EmotionResult("happy", 0.8))
        assert full_app.title == "😀 happy ⏱"

    def test_active_break_timer_clock_survives_icon_only_mode(self, full_app) -> None:
        full_app._break_timer["duration_seconds"] = 60
        full_app._show_emojis = False
        full_app._show_labels = False
        full_app._last_render = None
        full_app._render_emotion(EmotionResult("happy", 0.8))
        assert full_app.icon is not None
        assert full_app.title == "⏱"

    def test_set_menubar_skips_redundant_render(self, full_app) -> None:
        full_app._last_render = None
        full_app._set_menubar(None, "😀 happy")
        assert full_app.title == "😀 happy"
        # A second identical call is a no-op (state already cached).
        full_app.title = "sentinel"
        full_app._set_menubar(None, "😀 happy")
        assert full_app.title == "sentinel"


class TestToggles:
    def test_toggle_pause_round_trip(self, full_app) -> None:
        full_app.toggle_pause(None)
        assert full_app._paused is True
        assert "Resume" in full_app._pause_item.title
        full_app.toggle_pause(None)
        assert full_app._paused is False
        assert "Pause" in full_app._pause_item.title

    def test_toggle_emojis_persists(self, full_app, monkeypatch) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        full_app._paused = True  # skip the refresh branch
        full_app.toggle_emojis(full_app._emojis_item)
        assert full_app._show_emojis is False
        assert saved and saved[0]["show_emojis"] is False

    def test_toggle_labels_persists(self, full_app, monkeypatch) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        full_app._paused = True  # skip the refresh branch
        full_app.toggle_labels(full_app._labels_item)
        assert full_app._show_labels is False
        assert saved and saved[0]["show_labels"] is False


class _FakeSegment:
    """Minimal NSSegmentedControl stand-in for control-reading tests."""

    def __init__(self, selected: int) -> None:
        self._selected = selected

    def selectedSegment(self) -> int:
        return self._selected


class _FakeIntegerControl:
    def __init__(self, value: int) -> None:
        self._value = value

    def integerValue(self) -> int:
        return self._value


class _FakeToggle:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled

    def state(self) -> int:
        return int(self._enabled)


class TestNotifications:
    def test_defaults_disable_detected_emotions_only(self, full_app) -> None:
        assert full_app._notifications == app.DEFAULT_NOTIFICATIONS
        assert all(
            not full_app._notifications[event]
            for event in app.EMOTION_NOTIFICATION_EVENTS.values()
        )
        assert all(
            full_app._notifications[event]
            for event in app.NOTIFICATION_KEYS
            if not event.startswith("emotion_")
        )

    def test_multiple_faces_notification_is_in_detected_emotion_group(self) -> None:
        detected_options = next(
            options
            for group, options in app.NOTIFICATION_GROUPS
            if group == "Detected Emotion"
        )
        keys = [key for key, _label, _icon in detected_options]
        assert keys[-2:] == ["emotion_no_face", "emotion_multiple_faces"]
        assert app.DEFAULT_NOTIFICATIONS["emotion_multiple_faces"] is False

    def test_menu_item_precedes_sensitivity(self, full_app) -> None:
        menu_titles = list(full_app.menu.keys())
        sensitivity_index = menu_titles.index(full_app._sensitivity_menu.title)
        assert menu_titles[sensitivity_index - 1] == full_app._notifications_menu.title

    def test_apply_persists_each_toggle(self, full_app, monkeypatch) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda settings: saved.append(settings))
        notifications = dict.fromkeys(app.NOTIFICATION_KEYS, True)
        notifications["microphone_unmuted"] = False
        notifications["speakers_off"] = False
        assert full_app._apply_notifications(notifications) is True
        assert full_app._notifications == notifications
        assert saved[-1]["notifications"] == notifications

    def test_apply_from_controls_reads_each_toggle(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        controls = {
            key: _FakeToggle(index % 2 == 0)
            for index, key in enumerate(app.NOTIFICATION_KEYS)
        }
        assert full_app._apply_notifications_from_controls(controls) is True
        assert full_app._notifications == {
            key: index % 2 == 0
            for index, key in enumerate(app.NOTIFICATION_KEYS)
        }

    def test_build_accessory_restores_values_and_icons(self, full_app) -> None:
        full_app._notifications = {
            key: index % 2 == 0
            for index, key in enumerate(app.NOTIFICATION_KEYS)
        }
        _view, controls = full_app._build_notifications_accessory()
        assert set(controls) == set(app.NOTIFICATION_KEYS)
        for index, key in enumerate(app.NOTIFICATION_KEYS):
            assert bool(controls[key].state()) is (index % 2 == 0)
            assert controls[key].image() is not None

    def test_master_controls_update_all_toggles_and_summary(self, full_app) -> None:
        _view, controls = full_app._build_notifications_accessory()
        handler = full_app._notifications_handler
        handler.disableAll_(None)
        assert not any(bool(control.state()) for control in controls.values())
        assert handler.summary.stringValue().startswith("0 of ")
        handler.enableAll_(None)
        assert all(bool(control.state()) for control in controls.values())
        assert handler.summary.stringValue().startswith(
            f"{len(app.NOTIFICATION_KEYS)} of "
        )

    def test_load_restores_valid_values(self, data_dir, monkeypatch) -> None:
        app.save_settings(
            {
                "notifications": {
                    "microphone_muted": False,
                    "speakers_on": False,
                }
            }
        )
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        monkeypatch.setattr(app.rumps, "notification", lambda *args, **kwargs: None)
        inst = app.MooditoApp()
        assert inst._notifications["microphone_muted"] is False
        assert inst._notifications["speakers_on"] is False
        assert inst._notifications["microphone_unmuted"] is True
        assert inst._notifications["emotion_happy"] is False

    def test_saved_emotion_preference_overrides_default(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings({"notifications": {"emotion_happy": True}})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        monkeypatch.setattr(app.rumps, "notification", lambda *args, **kwargs: None)
        inst = app.MooditoApp()
        assert inst._notifications["emotion_happy"] is True
        assert inst._notifications["emotion_sad"] is False

    def test_disabled_event_does_not_notify(self, full_app, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            app.rumps, "notification", lambda *args, **kwargs: calls.append(args)
        )
        full_app._notifications["microphone_muted"] = False
        full_app._send_privacy_notification("microphone_muted")
        full_app._send_privacy_notification("speakers_off")
        assert calls == [
            ("Moodito Privacy", "Speakers volume off", "No face was detected.")
        ]

    def test_privacy_transition_sends_all_four_notifications(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"microphone_seconds": 1, "speakers_seconds": 1}
        )

        def mute(microphone, speakers):
            if microphone:
                state = app.AudioState(input_volume=70)
            else:
                state = app.AudioState(output_volume=45, output_muted=False)
            return app.PrivacyMuteResult(state, applied=True)

        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            mute,
        )
        monkeypatch.setattr(app, "restore_audio_state", lambda value: True)
        calls = []
        monkeypatch.setattr(
            app.rumps, "notification", lambda *args, **kwargs: calls.append(args)
        )

        full_app._update_privacy(False, now=1.0)
        full_app._update_privacy(False, now=2.0)
        full_app._update_privacy(True, now=3.0)

        assert [call[1] for call in calls] == [
            "Microphone muted",
            "Speakers volume off",
            "Microphone unmuted",
            "Speakers volume on",
        ]

    def test_multi_face_privacy_notifications_use_trigger_copy(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            multi_face={"microphone_seconds": 1}
        )
        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            lambda *_args: app.PrivacyMuteResult(
                app.AudioState(input_volume=70), applied=True
            ),
        )
        monkeypatch.setattr(app, "restore_audio_state", lambda _state: True)
        calls = []
        monkeypatch.setattr(
            app.rumps,
            "notification",
            lambda *args, **kwargs: calls.append(args),
        )

        full_app._update_privacy(app.MULTI_FACE_LABEL, now=1.0)
        full_app._update_privacy(app.MULTI_FACE_LABEL, now=2.0)
        full_app._update_privacy("happy", now=3.0)

        assert calls == [
            (
                "Moodito Privacy",
                "Microphone muted",
                "Multiple faces were detected.",
            ),
            (
                "Moodito Privacy",
                "Microphone unmuted",
                "Multiple faces are no longer detected.",
            ),
        ]

    def test_brightness_transition_sends_dimmed_and_restored_notifications(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"screen_brightness_seconds": 1}
        )
        brightness_states = (
            app.ScreenBrightnessState(display_id=42, level=0.75),
            app.ScreenBrightnessState(display_id=84, level=0.4),
        )
        monkeypatch.setattr(app, "dim_screens_for_privacy", lambda: brightness_states)
        monkeypatch.setattr(app, "restore_screen_brightness", lambda state: True)
        events = []
        monkeypatch.setattr(
            full_app,
            "_send_privacy_notification",
            lambda event: events.append(event),
        )

        full_app._update_privacy(False, now=1.0)
        full_app._update_privacy(False, now=2.0)
        full_app._update_privacy(True, now=3.0)

        assert events == ["brightness_dimmed", "brightness_restored"]

    def test_brightness_notifications_use_expected_messages(
        self, full_app, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            app.rumps,
            "notification",
            lambda *args, **kwargs: calls.append(args),
        )

        full_app._send_privacy_notification("brightness_dimmed")
        full_app._send_privacy_notification("brightness_restored")

        assert calls == [
            ("Moodito Privacy", "Brightness dimmed", "No face was detected."),
            (
                "Moodito Privacy",
                "Brightness restored",
                "Your face is visible again.",
            ),
        ]

    def test_already_dimmed_screen_does_not_notify(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"screen_brightness_seconds": 1}
        )
        brightness_states = (
            app.ScreenBrightnessState(display_id=42, level=0.0),
            app.ScreenBrightnessState(display_id=84, level=0.0),
        )
        monkeypatch.setattr(app, "dim_screens_for_privacy", lambda: brightness_states)
        monkeypatch.setattr(app, "restore_screen_brightness", lambda state: True)
        events = []
        monkeypatch.setattr(
            full_app,
            "_send_privacy_notification",
            lambda event: events.append(event),
        )

        full_app._update_privacy(False, now=1.0)
        full_app._update_privacy(False, now=2.0)
        full_app._update_privacy(True, now=3.0)

        assert events == []

    def test_already_silent_channels_do_not_notify(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"microphone_seconds": 1, "speakers_seconds": 1}
        )

        def mute(microphone, speakers):
            if microphone:
                state = app.AudioState(input_volume=0)
            else:
                state = app.AudioState(output_volume=0, output_muted=False)
            return app.PrivacyMuteResult(state, applied=True)

        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            mute,
        )
        monkeypatch.setattr(app, "restore_audio_state", lambda value: True)
        calls = []
        monkeypatch.setattr(
            app.rumps, "notification", lambda *args, **kwargs: calls.append(args)
        )

        full_app._update_privacy(False, now=1.0)
        full_app._update_privacy(False, now=2.0)
        full_app._update_privacy(True, now=3.0)

        assert calls == []

    def test_break_timer_notifications_have_independent_toggles(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        monkeypatch.setattr(full_app, "_show_break_timer_alert", lambda: None)
        full_app._notifications["break_timer_finished"] = False
        calls = []
        monkeypatch.setattr(
            app.rumps,
            "notification",
            lambda *args, **kwargs: calls.append(args),
        )

        full_app._apply_break_timer(10, 25)
        full_app._update_break_timer(True, now=0.0)
        full_app._update_break_timer(True, now=10.0)

        assert calls == [
            (
                "Moodito Break Timer",
                "Break Timer started",
                "Next break in 0h 0m 10s.",
            ),
            (
                "Moodito Break Timer",
                "Break Timer started",
                "Next break in 0h 0m 10s.",
            ),
        ]


class TestSensitivity:
    def test_defaults_to_normal_for_every_emotion(self, full_app) -> None:
        for emotion in app.SENSITIVITY_EMOTIONS:
            assert full_app._sensitivity[emotion] == app.DEFAULT_SENSITIVITY

    def test_menu_item_is_a_single_window_opener(self, full_app) -> None:
        # Sensitivity is a single item that opens the settings dialog.
        assert full_app._sensitivity_menu.title.startswith("Sensitivity")

    def test_worker_receives_initial_sensitivity(self, full_app) -> None:
        assert full_app._worker.sensitivity == full_app._sensitivity

    def test_apply_sensitivity_updates_worker_and_persists(
        self, full_app, monkeypatch
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        full_app._apply_sensitivity("happy", "high")
        assert full_app._sensitivity["happy"] == "high"
        assert full_app._worker.sensitivity["happy"] == "high"
        assert saved and saved[-1]["sensitivity"]["happy"] == "high"

    def test_apply_sensitivity_only_affects_its_emotion(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_sensitivity("angry", "low")
        assert full_app._sensitivity["angry"] == "low"
        assert full_app._sensitivity["happy"] == app.DEFAULT_SENSITIVITY

    def test_apply_sensitivity_ignores_invalid_input(
        self, full_app, monkeypatch
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        before = dict(full_app._sensitivity)
        full_app._apply_sensitivity("happy", "bogus")
        full_app._apply_sensitivity("nope", "high")
        assert full_app._sensitivity == before
        assert saved == []

    def test_apply_sensitivity_noop_when_unchanged(
        self, full_app, monkeypatch
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        # Already "normal" by default → no write.
        full_app._apply_sensitivity("happy", "normal")
        assert saved == []

    def test_apply_from_controls_commits_selected_levels(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        controls = {
            "happy": _FakeSegment(app.SENSITIVITY_LEVELS.index("high")),
            "angry": _FakeSegment(app.SENSITIVITY_LEVELS.index("low")),
        }
        full_app._apply_sensitivity_from_controls(controls)
        assert full_app._sensitivity["happy"] == "high"
        assert full_app._sensitivity["angry"] == "low"

    def test_apply_from_controls_ignores_out_of_range(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        before = dict(full_app._sensitivity)
        full_app._apply_sensitivity_from_controls({"happy": _FakeSegment(99)})
        assert full_app._sensitivity == before

    def test_build_accessory_has_a_control_per_emotion(self, full_app) -> None:
        _view, controls = full_app._build_sensitivity_accessory()
        assert set(controls) == set(app.SENSITIVITY_EMOTIONS)
        for emotion, control in controls.items():
            assert control.segmentCount() == len(app.SENSITIVITY_LEVELS)
            # Each control starts on the emotion's current level.
            assert control.selectedSegment() == app.SENSITIVITY_LEVELS.index(
                full_app._sensitivity[emotion]
            )

    def test_load_sensitivity_restores_valid_levels(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings({"sensitivity": {"happy": "high", "angry": "low"}})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._sensitivity["happy"] == "high"
        assert inst._sensitivity["angry"] == "low"
        # Untouched emotions keep the default.
        assert inst._sensitivity["sad"] == app.DEFAULT_SENSITIVITY

    def test_load_sensitivity_ignores_invalid_values(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings({"sensitivity": {"happy": "bogus", "sad": 5}})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._sensitivity["happy"] == app.DEFAULT_SENSITIVITY
        assert inst._sensitivity["sad"] == app.DEFAULT_SENSITIVITY


class TestPrivacy:
    def test_defaults_are_opt_in_for_both_triggers(self, full_app) -> None:
        assert full_app._privacy == _privacy_config()

    def test_brightness_action_precedes_screen_lock(self) -> None:
        assert app.PRIVACY_ACTIONS[-2:] == ("screen_brightness", "lock_screen")

    def test_menu_item_opens_window(self, full_app) -> None:
        assert full_app._privacy_menu.title.startswith("Privacy")

    def test_menu_item_follows_sensitivity(self, full_app) -> None:
        menu_titles = list(full_app.menu.keys())
        sensitivity_index = menu_titles.index(full_app._sensitivity_menu.title)
        assert menu_titles[sensitivity_index + 1] == full_app._privacy_menu.title

    def test_apply_persists_both_trigger_groups(self, full_app, monkeypatch) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda settings: saved.append(settings))
        privacy = {
            "no_face": {
                "microphone_seconds": 15,
                "speakers_seconds": 90,
                "screen_brightness_seconds": 120,
                "lock_screen_seconds": 300,
            },
            "multi_face": {
                "microphone_seconds": 5,
                "speakers_seconds": 10,
                "screen_brightness_seconds": 20,
                "lock_screen_seconds": 30,
            },
        }

        assert full_app._apply_privacy(privacy) is True
        assert full_app._privacy == privacy
        assert saved[-1]["privacy"] == privacy

    def test_apply_from_controls_reads_both_trigger_groups(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        controls = {
            f"{trigger}_{action}_{component}": _FakeIntegerControl(0)
            for trigger in app.PRIVACY_TRIGGERS
            for action in app.PRIVACY_ACTIONS
            for component in ("hours", "minutes", "seconds")
        }
        controls["no_face_microphone_minutes"] = _FakeIntegerControl(2)
        controls["multi_face_lock_screen_seconds"] = _FakeIntegerControl(45)

        assert full_app._apply_privacy_from_controls(controls) is True
        assert full_app._privacy["no_face"]["microphone_seconds"] == 120
        assert full_app._privacy["multi_face"]["lock_screen_seconds"] == 45

    def test_privacy_accessory_switches_trigger_panels(self, full_app) -> None:
        _view, controls = full_app._build_privacy_accessory()
        selector = controls["_trigger_selector"]
        panels = controls["_trigger_panels"]

        assert selector.segmentCount() == len(app.PRIVACY_TRIGGERS)
        assert selector.labelForSegment_(0) == "No Face"
        assert selector.labelForSegment_(1) == "Multiple Faces"
        assert panels[0].isHidden() is False
        assert panels[1].isHidden() is True

        selector.setSelectedSegment_(1)
        full_app._privacy_stepper_handler.triggerChanged_(selector)

        assert panels[0].isHidden() is True
        assert panels[1].isHidden() is False

    def test_apply_persists_settings(self, full_app, monkeypatch) -> None:
        saved = []
        notifications = []
        monkeypatch.setattr(app, "save_settings", lambda settings: saved.append(settings))
        monkeypatch.setattr(
            full_app,
            "_send_notification",
            lambda event, message=None: notifications.append((event, message)),
        )
        privacy = _privacy_config(
            no_face={
                "microphone_seconds": 15,
                "speakers_seconds": 90,
                "screen_brightness_seconds": 120,
                "lock_screen_seconds": 300,
            }
        )
        assert full_app._apply_privacy(privacy) is True
        assert full_app._privacy == privacy
        assert saved[-1]["privacy"] == full_app._privacy
        assert notifications == [
            (
                "privacy_settings_changed",
                "No Face: Microphone 0h 0m 15s, Speakers 0h 1m 30s, "
                "Brightness 0h 2m 0s, Lock screen 0h 5m 0s; "
                "Multiple Faces: Disabled.",
            )
        ]

    def test_apply_privacy_reports_disabled_zero_counter(
        self, full_app, monkeypatch
    ) -> None:
        notifications = []
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        monkeypatch.setattr(
            full_app,
            "_send_notification",
            lambda event, message=None: notifications.append((event, message)),
        )
        privacy = _privacy_config(no_face={"speakers_seconds": 30})
        assert full_app._apply_privacy(privacy) is True
        assert notifications == [
            (
                "privacy_settings_changed",
                "No Face: Speakers 0h 0m 30s; Multiple Faces: Disabled.",
            )
        ]

    def test_unchanged_or_failed_privacy_apply_does_not_notify(
        self, full_app, monkeypatch
    ) -> None:
        notifications = []
        monkeypatch.setattr(
            full_app,
            "_send_notification",
            lambda event, message=None: notifications.append((event, message)),
        )
        assert full_app._apply_privacy(_privacy_config()) is True
        assert full_app._apply_privacy(
            _privacy_config(no_face={"microphone_seconds": -1})
        ) is False
        full_app._privacy_audio_states = {
            "microphone": app.AudioState(input_volume=55)
        }
        monkeypatch.setattr(app, "restore_audio_state", lambda state: False)
        assert full_app._apply_privacy(
            _privacy_config(no_face={"microphone_seconds": 10})
        ) is False
        assert notifications == []

    @pytest.mark.parametrize(
        "microphone_seconds,speakers_seconds,screen_brightness_seconds,lock_screen_seconds",
        [
            (-1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, -1),
            (86400, 0, 0, 0),
            (0, 86400, 0, 0),
            (0, 0, 86400, 0),
            (0, 0, 0, 86400),
            (True, 1, 1, 1),
        ],
    )
    def test_apply_rejects_invalid_delay(
        self,
        full_app,
        monkeypatch,
        microphone_seconds,
        speakers_seconds,
        screen_brightness_seconds,
        lock_screen_seconds,
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda settings: saved.append(settings))
        privacy = _privacy_config(
            no_face={
                "microphone_seconds": microphone_seconds,
                "speakers_seconds": speakers_seconds,
                "screen_brightness_seconds": screen_brightness_seconds,
                "lock_screen_seconds": lock_screen_seconds,
            }
        )
        assert (
            full_app._apply_privacy(privacy)
            is False
        )
        assert saved == []

    def test_apply_from_controls_reads_all_values(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        controls = {
            f"{trigger}_{action}_{component}": _FakeIntegerControl(0)
            for trigger in app.PRIVACY_TRIGGERS
            for action in app.PRIVACY_ACTIONS
            for component in ("hours", "minutes", "seconds")
        }
        controls.update(
            {
                "no_face_microphone_hours": _FakeIntegerControl(1),
                "no_face_microphone_minutes": _FakeIntegerControl(2),
                "no_face_microphone_seconds": _FakeIntegerControl(3),
                "no_face_speakers_minutes": _FakeIntegerControl(59),
                "no_face_speakers_seconds": _FakeIntegerControl(59),
                "no_face_screen_brightness_hours": _FakeIntegerControl(4),
                "no_face_screen_brightness_minutes": _FakeIntegerControl(5),
                "no_face_screen_brightness_seconds": _FakeIntegerControl(6),
                "no_face_lock_screen_hours": _FakeIntegerControl(23),
                "no_face_lock_screen_minutes": _FakeIntegerControl(59),
                "no_face_lock_screen_seconds": _FakeIntegerControl(59),
            }
        )
        assert full_app._apply_privacy_from_controls(controls) is True
        assert full_app._privacy == _privacy_config(
            no_face={
                "microphone_seconds": 3723,
                "speakers_seconds": 3599,
                "screen_brightness_seconds": 14706,
                "lock_screen_seconds": 86399,
            }
        )

    @pytest.mark.parametrize(
        "key,value",
        [
            ("no_face_microphone_hours", 24),
            ("no_face_microphone_minutes", 60),
            ("multi_face_microphone_seconds", 60),
            ("multi_face_microphone_hours", -1),
        ],
    )
    def test_apply_from_controls_rejects_out_of_range_component(
        self, full_app, monkeypatch, key, value
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda settings: saved.append(settings))
        controls = {
            f"{trigger}_{action}_{component}": _FakeIntegerControl(0)
            for trigger in app.PRIVACY_TRIGGERS
            for action in app.PRIVACY_ACTIONS
            for component in ("hours", "minutes", "seconds")
        }
        controls[key] = _FakeIntegerControl(value)

        assert full_app._apply_privacy_from_controls(controls) is False
        assert saved == []

    def test_build_accessory_restores_current_values(self, full_app) -> None:
        full_app._privacy = _privacy_config(
            no_face={
                "microphone_seconds": 3723,
                "speakers_seconds": 3598,
                "screen_brightness_seconds": 7322,
                "lock_screen_seconds": 86399,
            },
            multi_face={"microphone_seconds": 45},
        )
        _view, controls = full_app._build_privacy_accessory()
        assert controls["no_face_microphone_hours"].integerValue() == 1
        assert controls["no_face_microphone_minutes"].integerValue() == 2
        assert controls["no_face_microphone_seconds"].integerValue() == 3
        assert controls["no_face_speakers_hours"].integerValue() == 0
        assert controls["no_face_speakers_minutes"].integerValue() == 59
        assert controls["no_face_speakers_seconds"].integerValue() == 58
        assert controls["no_face_screen_brightness_hours"].integerValue() == 2
        assert controls["no_face_screen_brightness_minutes"].integerValue() == 2
        assert controls["no_face_screen_brightness_seconds"].integerValue() == 2
        assert controls["no_face_lock_screen_hours"].integerValue() == 23
        assert controls["no_face_lock_screen_minutes"].integerValue() == 59
        assert controls["no_face_lock_screen_seconds"].integerValue() == 59
        assert controls["multi_face_microphone_seconds"].integerValue() == 45
        maximums = {
            "hours": app.MAX_PRIVACY_HOURS,
            "minutes": app.MAX_PRIVACY_MINUTES,
            "seconds": app.MAX_PRIVACY_COMPONENT_SECONDS,
        }
        for trigger in app.PRIVACY_TRIGGERS:
            for action in app.PRIVACY_ACTIONS:
                for component, maximum in maximums.items():
                    stepper = controls[
                        f"{trigger}_{action}_{component}_stepper"
                    ]
                    assert stepper.minValue() == 0
                    assert stepper.maxValue() == maximum

    def test_stepper_updates_counter_and_clamps_typed_value(self, full_app) -> None:
        _view, controls = full_app._build_privacy_accessory()
        field = controls["no_face_microphone_hours"]
        stepper = controls["no_face_microphone_hours_stepper"]
        stepper.setIntegerValue_(20)
        full_app._privacy_stepper_handler.stepperChanged_(stepper)
        assert field.integerValue() == 20
        field.setIntegerValue_(30)
        full_app._privacy_stepper_handler.fieldChanged_(field)
        assert field.integerValue() == 23
        assert stepper.integerValue() == 23

        minute_field = controls["multi_face_microphone_minutes"]
        minute_stepper = controls["multi_face_microphone_minutes_stepper"]
        minute_field.setIntegerValue_(70)
        full_app._privacy_stepper_handler.fieldChanged_(minute_field)
        assert minute_field.integerValue() == 59
        assert minute_stepper.integerValue() == 59

    def test_load_restores_per_channel_settings(self, data_dir, monkeypatch) -> None:
        app.save_settings(
            {
                "privacy": {
                    "microphone_seconds": 240,
                    "speakers_seconds": 20,
                    "screen_brightness_seconds": 75,
                    "lock_screen_seconds": 450,
                }
            }
        )
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._privacy == _privacy_config(
            no_face={
                "microphone_seconds": 240,
                "speakers_seconds": 20,
                "screen_brightness_seconds": 75,
                "lock_screen_seconds": 450,
            }
        )

    def test_load_restores_both_trigger_groups(self, data_dir, monkeypatch) -> None:
        privacy = _privacy_config(
            no_face={"microphone_seconds": 15},
            multi_face={"lock_screen_seconds": 45},
        )
        app.save_settings({"privacy": privacy})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)

        inst = app.MooditoApp()

        assert inst._privacy == privacy

    def test_multi_face_uses_its_own_action_delay(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = {
            "no_face": {
                "microphone_seconds": 0,
                "speakers_seconds": 0,
                "screen_brightness_seconds": 0,
                "lock_screen_seconds": 0,
            },
            "multi_face": {
                "microphone_seconds": 0,
                "speakers_seconds": 2,
                "screen_brightness_seconds": 0,
                "lock_screen_seconds": 0,
            },
        }
        calls = []
        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            lambda *args: calls.append(args)
            or app.PrivacyMuteResult(
                app.AudioState(output_volume=50, output_muted=False),
                applied=True,
            ),
        )

        full_app._update_privacy(app.MULTI_FACE_LABEL, now=10.0)
        full_app._update_privacy(app.MULTI_FACE_LABEL, now=11.9)
        assert calls == []
        full_app._update_privacy(app.MULTI_FACE_LABEL, now=12.0)

        assert calls == [(False, True)]

    def test_multi_face_can_activate_all_four_actions(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            multi_face={
                "microphone_seconds": 1,
                "speakers_seconds": 1,
                "screen_brightness_seconds": 1,
                "lock_screen_seconds": 1,
            }
        )
        calls = []

        def mute(microphone, speakers):
            calls.append(("mute", microphone, speakers))
            state = (
                app.AudioState(input_volume=70)
                if microphone
                else app.AudioState(output_volume=50, output_muted=False)
            )
            return app.PrivacyMuteResult(state, applied=True)

        monkeypatch.setattr(app, "mute_audio_for_privacy", mute)
        monkeypatch.setattr(
            app,
            "dim_screens_for_privacy",
            lambda: calls.append("dim")
            or (app.ScreenBrightnessState(display_id=1, level=0.5),),
        )
        monkeypatch.setattr(
            app,
            "lock_screen_for_privacy",
            lambda: calls.append("lock") or True,
        )
        monkeypatch.setattr(
            full_app,
            "_send_privacy_notification",
            lambda *_args: None,
        )

        full_app._update_privacy(app.MULTI_FACE_LABEL, now=1.0)
        full_app._update_privacy(app.MULTI_FACE_LABEL, now=2.0)

        assert calls == [
            ("mute", True, False),
            ("mute", False, True),
            "dim",
            "lock",
        ]

    def test_switching_triggers_restores_before_starting_new_delay(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"speakers_seconds": 1},
            multi_face={"speakers_seconds": 1},
        )
        calls = []

        def mute(_microphone, _speakers):
            calls.append("mute")
            return app.PrivacyMuteResult(
                app.AudioState(output_volume=50, output_muted=False),
                applied=True,
            )

        monkeypatch.setattr(app, "mute_audio_for_privacy", mute)
        monkeypatch.setattr(
            app,
            "restore_audio_state",
            lambda _state: calls.append("restore") or True,
        )
        monkeypatch.setattr(
            full_app,
            "_send_privacy_notification",
            lambda *_args: None,
        )

        full_app._update_privacy(app.NO_FACE_LABEL, now=1.0)
        full_app._update_privacy(app.NO_FACE_LABEL, now=2.0)
        full_app._update_privacy(app.MULTI_FACE_LABEL, now=3.0)
        assert calls == ["mute", "restore"]
        full_app._update_privacy(app.MULTI_FACE_LABEL, now=4.0)

        assert calls == ["mute", "restore", "mute"]

    def test_load_existing_settings_defaults_brightness_to_disabled(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings(
            {
                "privacy": {
                    "microphone_seconds": 240,
                    "speakers_seconds": 20,
                    "lock_screen_seconds": 450,
                }
            }
        )
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)

        inst = app.MooditoApp()
        assert inst._privacy == _privacy_config(
            no_face={
                "microphone_seconds": 240,
                "speakers_seconds": 20,
                "lock_screen_seconds": 450,
            }
        )

    def test_load_migrates_shared_delay_to_enabled_channels(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings(
            {
                "privacy": {
                    "minutes": 2,
                    "seconds": 5,
                    "microphone": True,
                    "speakers": False,
                }
            }
        )
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._privacy == _privacy_config(
            no_face={"microphone_seconds": 125}
        )

    def test_load_ignores_invalid_settings(self, data_dir, monkeypatch) -> None:
        app.save_settings(
            {
                "privacy": {
                    "minutes": -2,
                    "seconds": 80,
                    "microphone": "yes",
                    "speakers": 1,
                }
            }
        )
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._privacy == _privacy_config()

    def test_channels_activate_at_independent_delays(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"microphone_seconds": 5, "speakers_seconds": 10}
        )
        calls = []

        def mute(microphone, speakers):
            calls.append((microphone, speakers))
            if microphone:
                state = app.AudioState(input_volume=62)
            else:
                state = app.AudioState(output_volume=50, output_muted=False)
            return app.PrivacyMuteResult(state, applied=True)

        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            mute,
        )
        full_app._update_privacy(False, now=100.0)
        full_app._update_privacy(False, now=104.9)
        assert calls == []
        full_app._update_privacy(False, now=105.0)
        assert calls == [(True, False)]
        full_app._update_privacy(False, now=109.9)
        assert calls == [(True, False)]
        full_app._update_privacy(False, now=110.0)
        assert calls == [(True, False), (False, True)]
        assert set(full_app._privacy_audio_states) == {"microphone", "speakers"}

    def test_zero_disables_its_channel(self, full_app, monkeypatch) -> None:
        full_app._privacy = _privacy_config(
            no_face={"speakers_seconds": 5}
        )
        calls = []
        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            lambda *args: calls.append(args)
            or app.PrivacyMuteResult(
                app.AudioState(output_volume=50, output_muted=False), applied=True
            ),
        )
        full_app._update_privacy(False, now=1.0)
        full_app._update_privacy(False, now=6.0)
        assert calls == [(False, True)]

    def test_screen_locks_once_at_its_independent_delay(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"lock_screen_seconds": 5}
        )
        calls = []
        monkeypatch.setattr(
            app,
            "lock_screen_for_privacy",
            lambda: calls.append("lock") or True,
        )

        full_app._update_privacy(False, now=10.0)
        full_app._update_privacy(False, now=14.9)
        assert calls == []
        full_app._update_privacy(False, now=15.0)
        full_app._update_privacy(False, now=16.0)
        assert calls == ["lock"]

        full_app._update_privacy(True, now=17.0)
        full_app._update_privacy(False, now=20.0)
        full_app._update_privacy(False, now=25.0)
        assert calls == ["lock", "lock"]

    def test_screen_dims_before_lock_and_restores_when_face_returns(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={
                "screen_brightness_seconds": 5,
                "lock_screen_seconds": 5,
            }
        )
        full_app._privacy_brightness_states = ()
        brightness_states = (
            app.ScreenBrightnessState(display_id=42, level=0.75),
            app.ScreenBrightnessState(display_id=84, level=0.4),
        )
        calls = []
        monkeypatch.setattr(
            app,
            "dim_screens_for_privacy",
            lambda: calls.append("dim") or brightness_states,
        )
        monkeypatch.setattr(
            app,
            "lock_screen_for_privacy",
            lambda: calls.append("lock") or True,
        )
        monkeypatch.setattr(
            app,
            "restore_screen_brightness",
            lambda state: calls.append(("restore", state.display_id)) or True,
        )

        full_app._update_privacy(False, now=10.0)
        full_app._update_privacy(False, now=15.0)
        assert calls == ["dim", "lock"]

        full_app._update_privacy(True, now=16.0)
        assert calls == ["dim", "lock", ("restore", 42), ("restore", 84)]
        assert full_app._privacy_brightness_states == ()

    def test_failed_display_restore_is_retained_for_retry(
        self, full_app, monkeypatch
    ) -> None:
        first_state = app.ScreenBrightnessState(display_id=42, level=0.75)
        second_state = app.ScreenBrightnessState(display_id=84, level=0.4)
        full_app._privacy_brightness_states = (first_state, second_state)
        attempts = []

        def restore(state):
            attempts.append(state.display_id)
            return state.display_id == 42

        monkeypatch.setattr(app, "restore_screen_brightness", restore)

        assert full_app._restore_privacy_screen_brightness() is False
        assert attempts == [42, 84]
        assert full_app._privacy_brightness_states == (second_state,)

    def test_failed_reset_retains_trigger_context_for_restore_retry(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy_active_trigger = "multi_face"
        full_app._privacy_brightness_states = (
            app.ScreenBrightnessState(display_id=1, level=0.5),
        )
        results = iter((False, True))
        monkeypatch.setattr(
            app,
            "restore_screen_brightness",
            lambda _state: next(results),
        )
        notifications = []
        monkeypatch.setattr(
            full_app,
            "_send_privacy_notification",
            lambda *args: notifications.append(args),
        )

        assert full_app._reset_privacy() is False
        assert full_app._privacy_active_trigger == "multi_face"
        assert full_app._reset_privacy() is True

        assert full_app._privacy_active_trigger is None
        assert notifications == [
            (
                "brightness_restored",
                app.MULTI_FACE_ENDED_NOTIFICATION_MESSAGE,
            )
        ]

    def test_screen_lock_retries_after_failed_attempt(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"lock_screen_seconds": 1}
        )
        results = iter((False, True))
        calls = []
        monkeypatch.setattr(
            app,
            "lock_screen_for_privacy",
            lambda: calls.append("lock") or next(results),
        )

        full_app._update_privacy(False, now=1.0)
        full_app._update_privacy(False, now=2.0)
        assert "lock_screen" not in full_app._privacy_attempted["no_face"]
        full_app._update_privacy(False, now=2.3)

        assert calls == ["lock", "lock"]
        assert "lock_screen" in full_app._privacy_attempted["no_face"]

    def test_returning_face_restores_audio(self, full_app, monkeypatch) -> None:
        microphone_state = app.AudioState(input_volume=70)
        speakers_state = app.AudioState(output_volume=45, output_muted=False)
        full_app._privacy_audio_states = {
            "microphone": microphone_state,
            "speakers": speakers_state,
        }
        full_app._privacy_changed_channels = {"microphone", "speakers"}
        full_app._privacy_trigger_since["no_face"] = 10.0
        full_app._privacy_active_trigger = "no_face"
        restored = []
        monkeypatch.setattr(
            app, "restore_audio_state", lambda value: restored.append(value) or True
        )
        full_app._update_privacy(True, now=12.0)
        assert restored == [microphone_state, speakers_state]
        assert full_app._privacy_audio_states == {}
        assert full_app._privacy_trigger_since["no_face"] is None

    def test_face_interrupts_and_restarts_delay(self, full_app, monkeypatch) -> None:
        full_app._privacy = _privacy_config(
            no_face={"speakers_seconds": 5}
        )
        calls = []
        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            lambda *args: calls.append(args)
            or app.PrivacyMuteResult(
                app.AudioState(output_volume=50, output_muted=False), applied=True
            ),
        )
        full_app._update_privacy(False, now=20.0)
        full_app._update_privacy(True, now=24.0)
        full_app._update_privacy(False, now=25.0)
        full_app._update_privacy(False, now=29.9)
        assert calls == []
        full_app._update_privacy(False, now=30.0)
        assert calls == [(False, True)]

    def test_failed_mute_is_attempted_once_per_absence(
        self, full_app, monkeypatch
    ) -> None:
        full_app._privacy = _privacy_config(
            no_face={"microphone_seconds": 1}
        )
        calls = []
        monkeypatch.setattr(
            app,
            "mute_audio_for_privacy",
            lambda *args: calls.append(args) or None,
        )
        full_app._update_privacy(False, now=1.0)
        full_app._update_privacy(False, now=2.0)
        full_app._update_privacy(False, now=3.0)
        assert calls == [(True, False)]
        full_app._update_privacy(True, now=4.0)
        full_app._update_privacy(False, now=5.0)
        full_app._update_privacy(False, now=6.0)
        assert calls == [(True, False), (True, False)]


class TestBreakTimer:
    def test_defaults_are_disabled(self, full_app) -> None:
        assert full_app._break_timer == {
            "duration_seconds": 0,
            "absence_reset_percent": 0,
            "fired": False,
        }

    def test_menu_item_follows_privacy(self, full_app) -> None:
        menu_titles = list(full_app.menu.keys())
        privacy_index = menu_titles.index(full_app._privacy_menu.title)
        assert menu_titles[privacy_index + 1] == full_app._break_timer_menu.title

    def test_menu_clock_icon_is_always_shown(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        assert full_app._break_timer_menu._menuitem.image() is not None

        full_app._apply_break_timer(60, 20)
        assert full_app._break_timer_menu._menuitem.image() is not None

        full_app._apply_break_timer(0, 20)
        assert full_app._break_timer_menu._menuitem.image() is not None

    def test_menu_title_shows_remaining_hh_mm_ss(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        assert full_app._break_timer_menu.title == "Break Timer…"

        full_app._apply_break_timer(3661, 20)
        assert full_app._break_timer_menu.title == "Break Timer… [01:01:01]"
        full_app._update_break_timer(True, now=100.0)
        full_app._update_break_timer(True, now=100.1)
        assert full_app._break_timer_menu.title == "Break Timer… [01:01:01]"
        full_app._update_break_timer(True, now=101.0)
        assert full_app._break_timer_menu.title == "Break Timer… [01:01:00]"

        full_app._apply_break_timer(0, 20)
        assert full_app._break_timer_menu.title == "Break Timer…"

    def test_absence_reset_restores_full_menu_countdown(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        full_app._apply_break_timer(100, 20)
        full_app._update_break_timer(True, now=0.0)
        full_app._update_break_timer(True, now=50.0)
        assert full_app._break_timer_menu.title == "Break Timer… [00:00:50]"

        full_app._update_break_timer(False, now=60.0)
        full_app._update_break_timer(False, now=80.1)
        assert full_app._break_timer_menu.title == "Break Timer… [00:01:40]"

    def test_apply_from_controls_reads_duration_and_percentage(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        controls = {
            "hours": _FakeIntegerControl(1),
            "minutes": _FakeIntegerControl(2),
            "seconds": _FakeIntegerControl(3),
            "reset_percent": _FakeIntegerControl(99),
        }

        assert full_app._apply_break_timer_from_controls(controls) is True
        assert full_app._break_timer == {
            "duration_seconds": 3723,
            "absence_reset_percent": 99,
            "fired": False,
        }

    @pytest.mark.parametrize(
        "key,value",
        [
            ("hours", 24),
            ("minutes", 60),
            ("seconds", 60),
            ("reset_percent", 100),
            ("reset_percent", -1),
        ],
    )
    def test_apply_from_controls_rejects_invalid_component(
        self, full_app, monkeypatch, key, value
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda settings: saved.append(settings))
        controls = {
            "hours": _FakeIntegerControl(0),
            "minutes": _FakeIntegerControl(0),
            "seconds": _FakeIntegerControl(0),
            "reset_percent": _FakeIntegerControl(0),
        }
        controls[key] = _FakeIntegerControl(value)

        assert full_app._apply_break_timer_from_controls(controls) is False
        assert saved == []

    def test_accessory_restores_values_and_bounds(self, full_app) -> None:
        full_app._break_timer = {
            "duration_seconds": 86399,
            "absence_reset_percent": 99,
            "fired": False,
        }
        _view, controls = full_app._build_break_timer_accessory()

        assert controls["hours"].integerValue() == 23
        assert controls["minutes"].integerValue() == 59
        assert controls["seconds"].integerValue() == 59
        assert controls["reset_percent"].integerValue() == 99
        assert controls["hours_stepper"].maxValue() == 23
        assert controls["minutes_stepper"].maxValue() == 59
        assert controls["seconds_stepper"].maxValue() == 59
        assert controls["reset_percent_stepper"].maxValue() == 99

    def test_saved_finished_timer_restarts_after_reload(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings(
            {
                "break_timer": {
                    "duration_seconds": 60,
                    "absence_reset_percent": 20,
                    "fired": True,
                }
            }
        )
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        monkeypatch.setattr(app.rumps, "notification", lambda *args, **kwargs: None)
        inst = app.MooditoApp()
        alerts = []
        monkeypatch.setattr(inst, "_show_break_timer_alert", lambda: alerts.append(True))

        inst._update_break_timer(True, now=0.0)
        inst._update_break_timer(True, now=59.9)
        assert alerts == []
        inst._update_break_timer(True, now=60.0)

        assert inst._break_timer["fired"] is False
        assert alerts == [True]
        assert inst._break_timer_menu._menuitem.image() is not None

    def test_applying_finished_settings_explicitly_restarts_timer(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        full_app._break_timer = {
            "duration_seconds": 60,
            "absence_reset_percent": 20,
            "fired": True,
        }
        full_app._break_timer_elapsed = 60.0

        assert full_app._apply_break_timer(60, 20) is True
        assert full_app._break_timer["fired"] is False
        assert full_app._break_timer_elapsed == 0.0

    def test_finished_timer_opens_break_reminder(self, full_app, monkeypatch) -> None:
        import AppKit

        captured = {"buttons": [], "modal": False}

        class FakeAlert:
            @classmethod
            def alloc(cls):
                return cls()

            def init(self):
                return self

            def setMessageText_(self, value) -> None:
                captured["title"] = value

            def setInformativeText_(self, value) -> None:
                captured["message"] = value

            def addButtonWithTitle_(self, value) -> None:
                captured["buttons"].append(value)

            def setIcon_(self, _value) -> None:
                captured["icon"] = True

            def runModal(self) -> None:
                captured["modal"] = True

        class FakeImage:
            @classmethod
            def alloc(cls):
                return cls()

            def initByReferencingFile_(self, _path):
                return None

        fake_ns_app = type(
            "FakeNSApp",
            (),
            {"activateIgnoringOtherApps_": lambda self, _active: None},
        )()
        monkeypatch.setattr(AppKit, "NSAlert", FakeAlert)
        monkeypatch.setattr(AppKit, "NSImage", FakeImage)
        monkeypatch.setattr(AppKit, "NSApp", fake_ns_app)

        full_app._show_break_timer_alert()

        assert captured == {
            "buttons": ["OK"],
            "modal": True,
            "title": "Break Time",
            "message": (
                "Your break timer has finished. It is time to take a break."
            ),
        }

    def test_dismissing_break_reminder_starts_next_countdown(
        self, full_app, monkeypatch
    ) -> None:
        saved = []
        alerts = []
        monkeypatch.setattr(app, "save_settings", lambda settings: saved.append(settings))
        monkeypatch.setattr(
            full_app,
            "_show_break_timer_alert",
            lambda: alerts.append("break"),
        )

        assert full_app._apply_break_timer(10, 25) is True
        full_app._update_break_timer(True, now=100.0)
        full_app._update_break_timer(True, now=109.9)
        assert alerts == []
        full_app._update_break_timer(True, now=110.0)
        assert alerts == ["break"]
        assert full_app._break_timer["fired"] is False
        assert full_app._break_timer_elapsed == 0.0
        assert full_app._break_timer_menu.title == "Break Timer… [00:00:10]"

        full_app._update_break_timer(True, now=200.0)
        full_app._update_break_timer(True, now=209.9)
        assert alerts == ["break"]
        full_app._update_break_timer(True, now=210.0)

        assert alerts == ["break", "break"]
        assert full_app._break_timer == {
            "duration_seconds": 10,
            "absence_reset_percent": 25,
            "fired": False,
        }
        assert saved[-1]["break_timer"] == full_app._break_timer

    def test_countdown_sends_started_finished_started_notifications(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        monkeypatch.setattr(full_app, "_show_break_timer_alert", lambda: None)
        calls = []
        monkeypatch.setattr(
            app.rumps,
            "notification",
            lambda *args, **kwargs: calls.append(args),
        )

        full_app._apply_break_timer(10, 25)
        full_app._update_break_timer(True, now=100.0)
        full_app._update_break_timer(True, now=110.0)
        full_app._update_break_timer(True, now=111.0)

        assert [call[1] for call in calls] == [
            "Break Timer started",
            "Break Timer finished",
            "Break Timer started",
        ]

    def test_long_absence_resets_and_waits_for_face(
        self, full_app, monkeypatch
    ) -> None:
        alerts = []
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        monkeypatch.setattr(
            full_app,
            "_show_break_timer_alert",
            lambda: alerts.append("break"),
        )
        full_app._apply_break_timer(100, 20)

        full_app._update_break_timer(True, now=0.0)
        full_app._update_break_timer(True, now=50.0)
        full_app._update_break_timer(False, now=60.0)
        full_app._update_break_timer(False, now=80.1)
        assert full_app._break_timer_elapsed == 0.0
        assert full_app._break_timer_waiting_for_face is True

        full_app._update_break_timer(False, now=200.0)
        assert alerts == []
        full_app._update_break_timer(True, now=201.0)
        full_app._update_break_timer(True, now=300.9)
        assert alerts == []
        full_app._update_break_timer(True, now=301.0)
        assert alerts == ["break"]

    def test_zero_percent_resets_after_any_continuous_absence(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        full_app._apply_break_timer(100, 0)

        full_app._update_break_timer(True, now=0.0)
        full_app._update_break_timer(False, now=10.0)
        assert full_app._break_timer_waiting_for_face is False
        full_app._update_break_timer(False, now=10.1)
        assert full_app._break_timer_waiting_for_face is True
        assert full_app._break_timer_elapsed == 0.0

    def test_short_absence_does_not_reset_countdown(
        self, full_app, monkeypatch
    ) -> None:
        alerts = []
        monkeypatch.setattr(app, "save_settings", lambda settings: None)
        monkeypatch.setattr(
            full_app,
            "_show_break_timer_alert",
            lambda: alerts.append("break"),
        )
        full_app._apply_break_timer(100, 20)

        full_app._update_break_timer(True, now=0.0)
        full_app._update_break_timer(False, now=50.0)
        full_app._update_break_timer(False, now=69.9)
        full_app._update_break_timer(True, now=70.0)
        assert full_app._break_timer_waiting_for_face is False
        full_app._update_break_timer(True, now=100.0)

        assert alerts == ["break"]


class TestAIProvider:
    def test_defaults_to_default_provider_with_no_values(self, full_app) -> None:
        assert full_app._ai_provider["provider"] == app.DEFAULT_AI_PROVIDER
        assert full_app._ai_provider["providers"] == {}

    def test_menu_item_follows_mood_tip(self, full_app) -> None:
        menu_titles = list(full_app.menu.keys())
        mood_tip_index = menu_titles.index(full_app._mood_tip_menu.title)
        assert menu_titles[mood_tip_index + 1] == full_app._ai_provider_menu.title

    def test_menu_item_is_a_single_window_opener(self, full_app) -> None:
        assert "AI Provider" in full_app._ai_provider_menu.title

    def test_menu_title_shows_selected_provider(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        # Initial title reflects the default provider and marks it unconfigured.
        assert full_app._ai_provider_menu.title == (
            f"AI Provider: {app.DEFAULT_AI_PROVIDER} \u2717"
        )
        # Switching providers shows a ✓, the provider and its model (not the URL).
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt-4o"})
        assert full_app._ai_provider_menu.title == "AI Provider: OpenAI (gpt-4o) \u2713"

    def test_apply_provider_persists_only_relevant_fields(
        self, full_app, monkeypatch
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        full_app._apply_ai_provider(
            "Anthropic", {"api_key": "sk-123", "model": "claude", "url": "ignored"}
        )
        assert full_app._ai_provider["provider"] == "Anthropic"
        stored = full_app._ai_provider["providers"]["Anthropic"]
        assert stored == {"api_key": "sk-123", "model": "claude"}
        assert "url" not in stored  # Anthropic has no URL field
        assert saved and saved[-1]["ai_provider"]["provider"] == "Anthropic"

    def test_apply_provider_trims_whitespace(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider(
            "Ollama", {"url": "  http://localhost:11434  ", "model": " llama3 "}
        )
        stored = full_app._ai_provider["providers"]["Ollama"]
        assert stored == {"url": "http://localhost:11434", "model": "llama3"}

    def test_apply_provider_ignores_unknown_provider(
        self, full_app, monkeypatch
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        before = dict(full_app._ai_provider)
        full_app._apply_ai_provider("Nope", {"api_key": "x"})
        assert full_app._ai_provider == before
        assert saved == []

    def test_apply_provider_keeps_each_provider_config(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "sk-a", "model": "gpt"})
        full_app._apply_ai_provider(
            "OpenAI Compatible",
            {"url": "https://x", "api_key": "sk-b", "model": "m"},
        )
        # The earlier OpenAI config is still preserved.
        assert full_app._ai_provider["providers"]["OpenAI"]["api_key"] == "sk-a"
        assert full_app._ai_provider["provider"] == "OpenAI Compatible"

    def test_provider_values_returns_stored_fields(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("Gemini", {"api_key": "g-key", "model": "pro"})
        assert full_app._ai_provider_values("Gemini") == {
            "api_key": "g-key",
            "model": "pro",
        }

    def test_provider_values_empty_when_unset(self, full_app) -> None:
        assert full_app._ai_provider_values("OpenAI") == {
            "api_key": "",
            "model": "",
        }

    def test_load_provider_restores_valid_config(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings(
            {
                "ai_provider": {
                    "provider": "Ollama",
                    "providers": {
                        "Ollama": {"url": "http://host", "model": "llama"},
                        "Bogus": {"x": 1},  # unknown provider, dropped
                    },
                }
            }
        )
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._ai_provider["provider"] == "Ollama"
        assert inst._ai_provider["providers"]["Ollama"] == {
            "url": "http://host",
            "model": "llama",
        }
        assert "Bogus" not in inst._ai_provider["providers"]

    def test_load_provider_ignores_invalid_selection(
        self, data_dir, monkeypatch
    ) -> None:
        app.save_settings({"ai_provider": {"provider": "Nope"}})
        monkeypatch.setattr(app.FaceWorker, "start", lambda self: None)
        monkeypatch.setattr(app, "camera_authorization_status", lambda: 3)
        inst = app.MooditoApp()
        assert inst._ai_provider["provider"] == app.DEFAULT_AI_PROVIDER

    def test_build_provider_accessory_lists_all_providers(self, full_app) -> None:
        _view, popup = full_app._build_ai_provider_accessory()
        assert popup.numberOfItems() == len(app.AI_PROVIDERS)
        # The pop-up starts on the currently selected provider.
        assert popup.indexOfSelectedItem() == app.AI_PROVIDERS.index(
            full_app._ai_provider["provider"]
        )

    def test_build_fields_accessory_has_provider_fields(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider(
            "OpenAI Compatible",
            {"url": "https://x", "api_key": "sk", "model": "m"},
        )
        _view, fields, _status = full_app._build_ai_fields_accessory(
            "OpenAI Compatible"
        )
        assert set(fields) == set(app.AI_PROVIDER_FIELDS["OpenAI Compatible"])
        # Fields start pre-filled with their stored values.
        assert str(fields["url"].stringValue()) == "https://x"
        assert str(fields["model"].stringValue()) == "m"

    def test_apply_if_tested_commits_on_success(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        monkeypatch.setattr(
            app, "test_ai_connection", lambda *a, **k: (True, "Connection successful.")
        )
        status = _FakeStatusLabel()
        applied = full_app._apply_ai_provider_if_tested(
            "OpenAI", {"api_key": "k", "model": "gpt"}, status
        )
        assert applied is True
        assert status.value.startswith("✅")
        assert full_app._ai_provider["provider"] == "OpenAI"
        assert full_app._ai_provider["providers"]["OpenAI"]["model"] == "gpt"

    def test_apply_if_tested_blocks_on_failure(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        monkeypatch.setattr(
            app, "test_ai_connection", lambda *a, **k: (False, "invalid api key")
        )
        before_provider = full_app._ai_provider["provider"]
        before_providers = dict(full_app._ai_provider["providers"])
        status = _FakeStatusLabel()
        applied = full_app._apply_ai_provider_if_tested(
            "OpenAI", {"api_key": "bad", "model": "gpt"}, status
        )
        assert applied is False
        assert status.value.startswith("❌")
        assert "invalid api key" in status.value
        # Nothing was committed.
        assert full_app._ai_provider["provider"] == before_provider
        assert full_app._ai_provider["providers"] == before_providers

    def test_clear_removes_saved_credentials(self, full_app, monkeypatch) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})
        assert "OpenAI" in full_app._ai_provider["providers"]
        full_app._clear_ai_provider("OpenAI")
        assert "OpenAI" not in full_app._ai_provider["providers"]
        assert saved  # persisted

    def test_clear_unconfigured_provider_is_noop(
        self, full_app, monkeypatch
    ) -> None:
        saved = []
        monkeypatch.setattr(app, "save_settings", lambda s: saved.append(s))
        full_app._clear_ai_provider("Ollama")  # nothing stored; must not raise
        assert saved == []  # nothing to persist

    def test_clear_in_dialog_empties_fields_and_status(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})
        fields = {
            "api_key": _FakeTextField("k"),
            "model": _FakeTextField("gpt"),
        }
        status = _FakeStatusLabel()
        full_app._clear_ai_provider_in_dialog("OpenAI", fields, status)
        # Stored config is gone and the dialog fields are blanked.
        assert "OpenAI" not in full_app._ai_provider["providers"]
        assert all(f.stringValue() == "" for f in fields.values())
        assert "Cleared" in status.value
        # Title drops the model and marks it unconfigured now the config is gone.
        assert full_app._ai_provider_menu.title == "AI Provider: OpenAI \u2717"



class TestLLMConfigError:
    def test_unknown_provider(self) -> None:
        assert app.ai_provider_config_error("Nope", {}) == "unknown AI provider"

    def test_missing_api_key(self) -> None:
        assert (
            app.ai_provider_config_error("OpenAI", {"model": "gpt"})
            == "no API key configured"
        )

    def test_missing_url_for_compatible(self) -> None:
        assert (
            app.ai_provider_config_error(
                "OpenAI Compatible", {"api_key": "k", "model": "m"}
            )
            == "no URL configured"
        )

    def test_missing_model(self) -> None:
        assert (
            app.ai_provider_config_error("Anthropic", {"api_key": "k"})
            == "no model configured"
        )

    def test_ollama_needs_no_api_key(self) -> None:
        assert app.ai_provider_config_error(
            "Ollama", {"url": "http://h", "model": "m"}
        ) == ""

    def test_complete_config_is_valid(self) -> None:
        assert app.ai_provider_config_error(
            "OpenAI", {"api_key": "k", "model": "m"}
        ) == ""


class TestLLMHelpers:
    def test_openai_chat_url_appends_path(self) -> None:
        assert app._openai_chat_url("https://x/v1") == "https://x/v1/chat/completions"
        assert app._openai_chat_url("https://x/v1/") == "https://x/v1/chat/completions"

    def test_openai_chat_url_keeps_full_path(self) -> None:
        url = "https://x/v1/chat/completions"
        assert app._openai_chat_url(url) == url

    def test_ollama_chat_url_appends_path(self) -> None:
        assert app._ollama_chat_url("http://h:11434") == "http://h:11434/api/chat"
        assert app._ollama_chat_url("http://h:11434/") == "http://h:11434/api/chat"

    def test_error_detail_from_dict(self) -> None:
        assert app._llm_error_detail({"error": {"message": "boom"}}) == "boom"

    def test_error_detail_from_string(self) -> None:
        assert app._llm_error_detail({"error": "nope"}) == "nope"

    def test_error_detail_missing(self) -> None:
        assert app._llm_error_detail({}) == ""

    def test_llm_text_extracts_and_strips(self) -> None:
        data = {"choices": [{"message": {"content": "  hi  "}}]}
        assert app._llm_text(data, lambda d: d["choices"][0]["message"]["content"]) == "hi"

    def test_llm_text_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            app._llm_text({}, lambda d: d["choices"][0]["message"]["content"])

    def test_llm_text_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            app._llm_text({"t": "   "}, lambda d: d["t"])

    def test_format_hourly_durations_lists_nonzero_hours(self) -> None:
        values = [0.0] * 24
        values[9] = 120.0
        values[14] = 30.0
        out = app._format_hourly_durations(values)
        assert "09h" in out
        assert "14h" in out
        assert "00h" not in out

    def test_format_hourly_durations_empty_is_none(self) -> None:
        assert app._format_hourly_durations([0.0] * 24) == "none"

    def test_build_mood_report_prompt_includes_range_and_stats(self) -> None:
        start = app.datetime(2026, 6, 21, 8, 0)
        end = app.datetime(2026, 6, 21, 20, 0)
        stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
        stats["happy"] = {"seconds": 600.0, "count": 5}
        stats["sad"] = {"seconds": 200.0, "count": 2}
        hourly = [0.0] * 24
        hourly[9] = 400.0
        heat = {k: [0.0] * 24 for k in app.STAT_KEYS}
        heat["happy"][9] = 400.0
        prompt = app.build_mood_report_prompt(start, end, stats, hourly, heat)
        assert "Jun 21, 2026 08:00" in prompt
        assert "Jun 21, 2026 20:00" in prompt
        assert "happy" in prompt
        assert "occurrences" in prompt
        assert "09h" in prompt  # hourly detail present

    def test_build_mood_report_prompt_handles_live_end(self) -> None:
        start = app.datetime(2026, 6, 21, 8, 0)
        stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
        prompt = app.build_mood_report_prompt(
            start, None, stats, [0.0] * 24, {k: [0.0] * 24 for k in app.STAT_KEYS}
        )
        assert "to now" in prompt

    def test_build_mood_report_prompt_includes_activity(self) -> None:
        start = app.datetime(2026, 6, 21, 8, 0)
        end = app.datetime(2026, 6, 21, 20, 0)
        stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
        hourly = [0.0] * 24
        heat = {k: [0.0] * 24 for k in app.STAT_KEYS}
        prompt = app.build_mood_report_prompt(
            start, end, stats, hourly, heat, activity="Working on a presentation"
        )
        assert "Working on a presentation" in prompt
        assert "activity" in prompt.lower()

    def test_build_mood_report_prompt_omits_activity_when_empty(self) -> None:
        start = app.datetime(2026, 6, 21, 8, 0)
        stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
        prompt = app.build_mood_report_prompt(
            start, None, stats, [0.0] * 24, {k: [0.0] * 24 for k in app.STAT_KEYS}
        )
        assert "described their activity" not in prompt


class TestCallLLM:
    @staticmethod
    def _capture(monkeypatch, response):
        calls = {}

        def fake_post(url, payload, headers=None):
            calls["url"] = url
            calls["payload"] = payload
            calls["headers"] = headers or {}
            return response

        monkeypatch.setattr(app, "_llm_post_json", fake_post)
        return calls

    def test_anthropic_request_and_reply(self, monkeypatch) -> None:
        calls = self._capture(monkeypatch, {"content": [{"text": "be kind"}]})
        out = app.call_llm("Anthropic", {"api_key": "k", "model": "claude"}, "hi")
        assert out == "be kind"
        assert calls["url"] == app.ANTHROPIC_API_URL
        assert calls["headers"]["x-api-key"] == "k"
        assert calls["headers"]["anthropic-version"] == app.ANTHROPIC_VERSION
        assert calls["payload"]["model"] == "claude"

    def test_openai_request_and_reply(self, monkeypatch) -> None:
        calls = self._capture(
            monkeypatch, {"choices": [{"message": {"content": "smile"}}]}
        )
        out = app.call_llm("OpenAI", {"api_key": "k", "model": "gpt"}, "hi")
        assert out == "smile"
        assert calls["url"] == app.OPENAI_API_URL
        assert calls["headers"]["Authorization"] == "Bearer k"

    def test_openai_compatible_uses_custom_url(self, monkeypatch) -> None:
        calls = self._capture(
            monkeypatch, {"choices": [{"message": {"content": "ok"}}]}
        )
        app.call_llm(
            "OpenAI Compatible",
            {"url": "https://host/v1", "api_key": "k", "model": "m"},
            "hi",
        )
        assert calls["url"] == "https://host/v1/chat/completions"

    def test_openai_falls_back_to_max_completion_tokens(self, monkeypatch) -> None:
        attempts = []

        def fake_post(url, payload, headers=None):
            token_param = "max_tokens" if "max_tokens" in payload else (
                "max_completion_tokens"
            )
            attempts.append(token_param)
            if token_param == "max_tokens":
                raise ValueError(
                    "Unsupported parameter: 'max_tokens' is not supported with this "
                    "model. Use 'max_completion_tokens' instead"
                )
            return {"choices": [{"message": {"content": "done"}}]}

        monkeypatch.setattr(app, "_llm_post_json", fake_post)
        out = app.call_llm("OpenAI", {"api_key": "k", "model": "gpt-5"}, "hi")
        assert out == "done"
        assert attempts == ["max_tokens", "max_completion_tokens"]

    def test_openai_other_errors_are_not_retried(self, monkeypatch) -> None:
        attempts = []

        def fake_post(url, payload, headers=None):
            attempts.append(payload)
            raise ValueError("invalid api key")

        monkeypatch.setattr(app, "_llm_post_json", fake_post)
        with pytest.raises(ValueError, match="invalid api key"):
            app.call_llm("OpenAI", {"api_key": "k", "model": "gpt"}, "hi")
        assert len(attempts) == 1  # no retry on unrelated errors

    def test_gemini_request_and_reply(self, monkeypatch) -> None:
        calls = self._capture(
            monkeypatch,
            {"candidates": [{"content": {"parts": [{"text": "breathe"}]}}]},
        )
        out = app.call_llm("Gemini", {"api_key": "k", "model": "pro"}, "hi")
        assert out == "breathe"
        assert "models/pro:generateContent" in calls["url"]
        assert "key=k" in calls["url"]

    def test_ollama_request_and_reply(self, monkeypatch) -> None:
        calls = self._capture(monkeypatch, {"message": {"content": "rest"}})
        out = app.call_llm(
            "Ollama", {"url": "http://h:11434", "model": "llama"}, "hi"
        )
        assert out == "rest"
        assert calls["url"] == "http://h:11434/api/chat"

    def test_incomplete_config_raises(self, monkeypatch) -> None:
        self._capture(monkeypatch, {})
        with pytest.raises(ValueError):
            app.call_llm("OpenAI", {"model": "gpt"}, "hi")


class _FakeTextField:
    """Minimal NSTextField stand-in exposing get/set stringValue()."""

    def __init__(self, value: str = "") -> None:
        self._value = value

    def stringValue(self) -> str:
        return self._value

    def setStringValue_(self, value: str) -> None:
        self._value = value


class _FakeStatusLabel:
    """Minimal status-label stand-in recording set string/tooltip."""

    def __init__(self) -> None:
        self.value = ""
        self.tooltip = ""

    def setStringValue_(self, value: str) -> None:
        self.value = value

    def setToolTip_(self, value: str) -> None:
        self.tooltip = value


class _FakeTextView:
    """Minimal NSTextView stand-in recording the last setString_ value."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def setString_(self, value: str) -> None:
        self.value = value

    def string(self) -> str:
        return self.value


class _FakeButton:
    """Minimal NSButton stand-in recording its enabled state."""

    def __init__(self) -> None:
        self.enabled = True

    def setEnabled_(self, value) -> None:
        self.enabled = bool(value)


class _FakeMoodHandler:
    """Stand-in for the Mood Tip ObjC handler used in tests.

    Holds the fake text view/button and forwards the (normally main-thread)
    result delivery straight to the app, synchronously.
    """

    def __init__(self, app_obj) -> None:
        self.app = app_obj
        self.text_view = _FakeTextView()
        self.activity_view = _FakeTextView()
        self.button = _FakeButton()
        self.pdf_button = _FakeButton()
        self.pdf_button.enabled = False
        self.report_text = ""
        self.report_meta = None

    def showResult_(self, result) -> None:
        self.app._finish_mood_report(self, result)

    def performSelectorOnMainThread_withObject_waitUntilDone_modes_(
        self, _selector, obj, _wait, _modes
    ) -> None:
        self.showResult_(obj)


class TestAIConnectionTest:
    def test_success(self, monkeypatch) -> None:
        monkeypatch.setattr(app, "call_llm", lambda *a, **k: "OK")
        ok, message = app.test_ai_connection(
            "OpenAI", {"api_key": "k", "model": "gpt"}
        )
        assert ok is True
        assert "success" in message.lower()

    def test_incomplete_config_fails_without_network(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(app, "call_llm", lambda *a, **k: called.append(a))
        ok, message = app.test_ai_connection("OpenAI", {"model": "gpt"})
        assert ok is False
        assert message == "no API key configured"
        assert called == []  # never attempted a request

    def test_api_error_is_reported(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise ValueError("invalid api key")

        monkeypatch.setattr(app, "call_llm", boom)
        ok, message = app.test_ai_connection(
            "OpenAI", {"api_key": "bad", "model": "gpt"}
        )
        assert ok is False
        assert "invalid api key" in message

    def test_network_error_is_reported(self, monkeypatch) -> None:
        def offline(*a, **k):
            raise OSError("unreachable")

        monkeypatch.setattr(app, "call_llm", offline)
        ok, message = app.test_ai_connection(
            "Ollama", {"url": "http://h", "model": "m"}
        )
        assert ok is False
        assert "unreachable" in message

    def test_run_connection_test_updates_status_on_success(
        self, full_app, monkeypatch
    ) -> None:
        captured = {}

        def fake_test(provider, values):
            captured["provider"] = provider
            captured["values"] = values
            return True, "Connection successful."

        monkeypatch.setattr(app, "test_ai_connection", fake_test)
        status = _FakeStatusLabel()
        fields = {"api_key": _FakeTextField("sk-1"), "model": _FakeTextField("gpt")}
        full_app._run_ai_connection_test("OpenAI", fields, status)
        assert status.value.startswith("✅")
        assert "successful" in status.tooltip
        # The currently typed field values are what gets tested.
        assert captured["provider"] == "OpenAI"
        assert captured["values"] == {"api_key": "sk-1", "model": "gpt"}

    def test_run_connection_test_shows_error_status(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            app, "test_ai_connection", lambda *a, **k: (False, "invalid api key")
        )
        status = _FakeStatusLabel()
        full_app._run_ai_connection_test(
            "OpenAI", {"api_key": _FakeTextField("bad")}, status
        )
        assert status.value.startswith("❌")
        assert "invalid api key" in status.value
        assert status.tooltip == "invalid api key"


class TestMoodTip:
    def test_unconfigured_writes_message_to_window(self, full_app) -> None:
        # Default provider has no credentials, so generating reports the issue
        # in the window itself (the window already opened) and stays idle.
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        assert "set up your AI provider" in handler.text_view.value
        assert not full_app._llm_busy.is_set()

    def test_busy_guard_blocks_second_call(self, full_app, monkeypatch) -> None:
        started = []
        monkeypatch.setattr(
            app.threading, "Thread", lambda *a, **k: started.append(k)
        )
        full_app._llm_busy.set()
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        assert started == []  # no new thread while busy
        assert handler.text_view.value == ""  # window left untouched

    def test_generate_shows_report_in_window(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})
        monkeypatch.setattr(app, "call_llm", lambda *a, **k: "you've got this")
        _patch_sync_threads(monkeypatch)
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        assert not full_app._llm_busy.is_set()
        assert handler.text_view.value == "you've got this"
        assert handler.button.enabled is True

    def test_wait_message_shown_before_network_call(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})
        seen = {}

        def fake_call(provider, config, prompt):
            # While the network call runs, the window shows the wait message
            # and the button is disabled.
            seen["text"] = handler.text_view.value
            seen["enabled"] = handler.button.enabled
            return "done"

        monkeypatch.setattr(app, "call_llm", fake_call)
        _patch_sync_threads(monkeypatch)
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        assert seen["text"] == app.MOOD_TIP_WAIT
        assert seen["enabled"] is False

    def test_error_is_shown_in_window(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})

        def boom(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(app, "call_llm", boom)
        _patch_sync_threads(monkeypatch)
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        assert "Could not get a mood report" in handler.text_view.value
        assert "offline" in handler.text_view.value
        assert handler.button.enabled is True
        assert not full_app._llm_busy.is_set()

    def test_sends_range_report_prompt_to_llm(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})
        captured = {}

        def fake_call(provider, config, prompt):
            captured["prompt"] = prompt
            return "report"

        monkeypatch.setattr(app, "call_llm", fake_call)
        _patch_sync_threads(monkeypatch)
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        # The prompt is built from the selected range's aggregated data.
        assert "Total tracked time" in captured["prompt"]
        assert "wellbeing suggestions" in captured["prompt"]

    def test_build_accessory_wires_button_handler(self, full_app) -> None:
        # Smoke test the real AppKit accessory: it builds and retains a handler
        # wired to the app and its report text view / button.
        view = full_app._build_mood_tip_accessory()
        assert view is not None
        handler = full_app._mood_tip_handler
        assert handler is not None
        assert handler.app is full_app
        assert handler.text_view is not None
        assert handler.button is not None
        # The PDF export button exists and starts disabled (no report yet).
        assert handler.pdf_button is not None
        assert handler.pdf_button.isEnabled() is False

    def test_successful_report_enables_pdf_button(
        self, full_app, monkeypatch
    ) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})
        monkeypatch.setattr(app, "call_llm", lambda *a, **k: "stay positive")
        _patch_sync_threads(monkeypatch)
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        assert handler.report_text == "stay positive"
        assert handler.pdf_button.enabled is True
        # The report's metadata snapshot is captured for the PDF header.
        assert handler.report_meta is not None
        assert handler.report_meta["provider"] == "OpenAI"
        assert handler.report_meta["model"] == "gpt"

    def test_error_keeps_pdf_button_disabled(self, full_app, monkeypatch) -> None:
        monkeypatch.setattr(app, "save_settings", lambda s: None)
        full_app._apply_ai_provider("OpenAI", {"api_key": "k", "model": "gpt"})

        def boom(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(app, "call_llm", boom)
        _patch_sync_threads(monkeypatch)
        handler = _FakeMoodHandler(full_app)
        full_app._start_mood_report(handler)
        assert handler.report_text == ""
        assert handler.report_meta is None
        assert handler.pdf_button.enabled is False

    def test_save_pdf_no_report_is_noop(self, full_app, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(app, "write_report_pdf", lambda *a, **k: called.append(a))
        handler = _FakeMoodHandler(full_app)
        handler.report_text = ""
        full_app._save_mood_report_pdf(handler)
        assert called == []  # nothing to save, no panel, no write

    def test_collect_report_meta_summarises_range(self, full_app) -> None:
        # Seed a couple of emotions into the selected range and snapshot them.
        full_app._range_stats = {key: {"seconds": 0.0, "count": 0} for key in app.STAT_KEYS}
        full_app._range_stats["happy"] = {"seconds": 1800.0, "count": 5}
        full_app._range_stats["sad"] = {"seconds": 600.0, "count": 2}
        meta = full_app._collect_report_meta("OpenAI", {"model": "gpt-4o"})
        assert meta["provider"] == "OpenAI"
        assert meta["model"] == "gpt-4o"
        assert meta["emotion_count"] == 2  # happy + sad have occurrences
        keys = {row["key"]: row for row in meta["emotions"]}
        # Every tracked state is present, even those with zero occurrences.
        assert set(keys) == set(app.STAT_KEYS)
        assert keys["happy"]["count"] == 5
        assert keys["happy"]["pct"] == 75.0  # 1800 of 2400 total seconds
        assert keys["surprised"]["count"] == 0  # zero-occurrence emotion shown
        assert "duration" in keys["happy"]
        # Totals row sums the occurrences and accounts for all of the time.
        assert meta["totals"]["count"] == 7
        assert meta["totals"]["pct"] == 100.0
        assert "generated" in meta and meta["range_end"]

    def test_build_report_pdf_attributed_string_smoke(self) -> None:
        # The real AppKit builder produces a non-empty attributed string that
        # includes the report body and header labels.
        meta = {
            "generated": "Jun 27, 2026 10:00",
            "range_start": "Jun 26, 2026 10:00",
            "range_end": "Now",
            "total_duration": "2h 00m",
            "emotion_count": 1,
            "emotions": [
                {"key": "happy", "emoji": "😀", "name": "Happy",
                 "count": 3, "pct": 100.0, "duration": "2h 00m"},
            ],
            "provider": "OpenAI",
            "model": "gpt-4o",
        }
        attributed = app._build_report_attributed_string("Body text.", meta, 468.0)
        assert attributed.length() > 0
        plain = attributed.string()
        assert "Mood Report" in plain
        assert "Emotional Breakdown" in plain
        assert "Body text." in plain



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
    def __init__(self, face_count=1) -> None:
        self.face_blendshapes = [[_Blendshape()] for _ in range(face_count)]


class TestFaceWorkerRun:
    def test_run_reports_model_download_failure(self, monkeypatch) -> None:
        def boom() -> None:
            raise RuntimeError("no network")

        monkeypatch.setattr(app, "ensure_model", boom)
        worker = app.FaceWorker()
        worker.run()
        assert "model download failed" in worker.error

    def test_run_redownloads_model_after_initialization_failure(
        self, data_dir, monkeypatch
    ) -> None:
        model_path = data_dir / "model.task"
        model_path.write_bytes(b"invalid")
        ensure_calls = []

        def fake_ensure_model() -> None:
            ensure_calls.append(True)
            if not model_path.exists():
                model_path.write_bytes(b"fresh")

        class FakeLandmarkerContext:
            def __enter__(self):
                return object()

            def __exit__(self, *_args):
                return False

        def fake_create(_options):
            if model_path.read_bytes() == b"invalid":
                raise RuntimeError("invalid model")
            return FakeLandmarkerContext()

        monkeypatch.setattr(app, "ensure_model", fake_ensure_model)
        monkeypatch.setattr(
            app.vision,
            "FaceLandmarkerOptions",
            lambda **options: options,
        )
        monkeypatch.setattr(
            app.vision.FaceLandmarker,
            "create_from_options",
            fake_create,
        )
        worker = app.FaceWorker()
        worker.stop()

        worker.run()

        assert ensure_calls == [True, True]
        assert model_path.read_bytes() == b"fresh"
        assert worker.error is None

    def test_landmarker_is_configured_for_multiple_faces(self, monkeypatch) -> None:
        captured = {}
        monkeypatch.setattr(app, "ensure_model", lambda: None)
        monkeypatch.setattr(
            app.vision,
            "FaceLandmarkerOptions",
            lambda **options: captured.update(options) or options,
        )

        class FakeLandmarkerContext:
            def __enter__(self):
                return object()

            def __exit__(self, *_args):
                return False

        fake_landmarker = type(
            "FakeFaceLandmarker",
            (),
            {
                "create_from_options": staticmethod(
                    lambda _options: FakeLandmarkerContext()
                )
            },
        )
        monkeypatch.setattr(app.vision, "FaceLandmarker", fake_landmarker)
        worker = app.FaceWorker()
        worker.stop()

        worker.run()

        assert captured["num_faces"] == app.MAX_DETECTED_FACES
        assert captured["num_faces"] > 1

    def test_capture_loop_processes_face_frame(self, monkeypatch) -> None:
        monkeypatch.setattr(app.cv2, "cvtColor", lambda *a, **k: "rgb")
        monkeypatch.setattr(app.cv2, "COLOR_BGR2RGB", 0, raising=False)
        monkeypatch.setattr(app.mp, "Image", lambda **k: "image")
        monkeypatch.setattr(
            app, "infer_emotion", lambda scores, sensitivity=None: EmotionResult("happy", 0.9)
        )
        worker = app.FaceWorker()
        landmarker = type(
            "L", (), {"detect_for_video": lambda self, img, ts: _Detection(1)}
        )()
        cap = _FakeCap(worker, frames=2)
        worker._run_capture_loop(cap, landmarker)
        assert worker.result == EmotionResult("happy", 0.9)

    def test_capture_loop_preserves_timestamp_across_reconnects(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(app.cv2, "cvtColor", lambda *a, **k: "rgb")
        monkeypatch.setattr(app.cv2, "COLOR_BGR2RGB", 0, raising=False)
        monkeypatch.setattr(app.mp, "Image", lambda **k: "image")
        timestamps = []
        worker = app.FaceWorker()
        landmarker = type(
            "L",
            (),
            {
                "detect_for_video": lambda self, img, ts: (
                    timestamps.append(ts) or _Detection(0)
                )
            },
        )()

        worker._run_capture_loop(_FakeCap(worker, frames=1), landmarker)
        worker._stop.clear()
        worker._run_capture_loop(_FakeCap(worker, frames=1), landmarker)

        assert timestamps == [150, 300]

    def test_capture_loop_handles_no_face(self, monkeypatch) -> None:
        monkeypatch.setattr(app.cv2, "cvtColor", lambda *a, **k: "rgb")
        monkeypatch.setattr(app.cv2, "COLOR_BGR2RGB", 0, raising=False)
        monkeypatch.setattr(app.mp, "Image", lambda **k: "image")
        worker = app.FaceWorker()
        landmarker = type(
            "L", (), {"detect_for_video": lambda self, img, ts: _Detection(0)}
        )()
        cap = _FakeCap(worker, frames=2)
        worker._run_capture_loop(cap, landmarker)
        assert worker.result == EmotionResult("no face", 0.0, face_count=0)

    def test_capture_loop_reports_multiple_faces_without_emotion_inference(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(app.cv2, "cvtColor", lambda *a, **k: "rgb")
        monkeypatch.setattr(app.cv2, "COLOR_BGR2RGB", 0, raising=False)
        monkeypatch.setattr(app.mp, "Image", lambda **k: "image")
        infer_calls = []
        monkeypatch.setattr(
            app,
            "infer_emotion",
            lambda *args, **kwargs: infer_calls.append((args, kwargs)),
        )
        worker = app.FaceWorker()
        landmarker = type(
            "L", (), {"detect_for_video": lambda self, img, ts: _Detection(3)}
        )()
        cap = _FakeCap(worker, frames=2)

        worker._run_capture_loop(cap, landmarker)

        assert worker.result == EmotionResult(
            app.MULTI_FACE_LABEL,
            1.0,
            face_count=3,
        )
        assert infer_calls == []

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


def _capture_notifications(inst, monkeypatch, *enabled_events):
    """Enable selected events and capture macOS notification arguments."""
    inst._notifications = dict.fromkeys(app.NOTIFICATION_KEYS, False)
    for event in enabled_events:
        inst._notifications[event] = True
    calls = []
    monkeypatch.setattr(
        app.rumps, "notification", lambda *args, **kwargs: calls.append(args)
    )
    return calls


class TestNotificationClickDetails:
    def test_sent_notification_includes_timestamp_and_show_action(
        self, full_app, monkeypatch
    ) -> None:
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(app.rumps, "notification", capture)
        full_app._notifications["app_paused"] = True

        full_app._send_notification("app_paused")

        assert captured["kwargs"]["action_button"] == "Show"
        payload = captured["kwargs"]["data"]
        assert payload["event"] == "app_paused"
        assert payload["subtitle"] == "Moodito paused"
        assert app.parse_iso_datetime(payload["occurred_at"]) is not None

    def test_click_details_prefer_occurrence_payload_and_fallback_to_delivery(
        self,
    ) -> None:
        occurred = "2026-07-20T14:35:42"
        notification = type(
            "Notification",
            (),
            {
                "data": {
                    "subtitle": "Moodito paused",
                    "message": "Face detection is paused.",
                    "occurred_at": occurred,
                },
                "subtitle": "Fallback subtitle",
                "message": "Fallback message",
                "delivered_at": app.datetime(2026, 1, 1),
            },
        )()
        subtitle, message, occurred_at = app.MooditoApp._notification_detail_values(
            notification
        )
        assert subtitle == "Moodito paused"
        assert message == "Face detection is paused."
        assert occurred_at == app.datetime(2026, 7, 20, 14, 35, 42)

        notification.data = None
        _subtitle, _message, occurred_at = (
            app.MooditoApp._notification_detail_values(notification)
        )
        assert occurred_at == notification.delivered_at

    def test_show_action_opens_timestamp_window(self, full_app, monkeypatch) -> None:
        import AppKit

        captured = {"buttons": [], "modal": False}

        class FakeAlert:
            @classmethod
            def alloc(cls):
                return cls()

            def init(self):
                return self

            def setMessageText_(self, value) -> None:
                captured["title"] = value

            def setInformativeText_(self, value) -> None:
                captured["info"] = value

            def addButtonWithTitle_(self, value) -> None:
                captured["buttons"].append(value)

            def setIcon_(self, _value) -> None:
                pass

            def runModal(self) -> None:
                captured["modal"] = True

        class FakeImage:
            @classmethod
            def alloc(cls):
                return cls()

            def initByReferencingFile_(self, _path):
                return self

            def isValid(self) -> bool:
                return True

        fake_ns_app = type(
            "FakeNSApp",
            (),
            {"activateIgnoringOtherApps_": lambda self, _active: None},
        )()
        monkeypatch.setattr(AppKit, "NSAlert", FakeAlert)
        monkeypatch.setattr(AppKit, "NSImage", FakeImage)
        monkeypatch.setattr(AppKit, "NSApp", fake_ns_app)
        notification = type(
            "Notification",
            (),
            {
                "data": {
                    "subtitle": "CSV downloaded",
                    "message": "Saved moodito.csv in Downloads.",
                    "occurred_at": "2026-07-20T14:35:42",
                },
                "subtitle": "",
                "message": "",
                "delivered_at": None,
            },
        )()

        full_app._show_notification_details(notification)

        assert captured["title"] == "CSV downloaded"
        assert "Saved moodito.csv in Downloads." in captured["info"]
        assert "Jul 20, 2026 at 14:35:42" in captured["info"]
        assert captured["buttons"] == ["Close"]
        assert captured["modal"] is True

    def test_global_click_handler_routes_to_running_app(
        self, full_app, monkeypatch
    ) -> None:
        received = []
        monkeypatch.setattr(
            full_app,
            "_show_notification_details",
            lambda notification: received.append(notification),
        )
        monkeypatch.setattr(
            app.rumps.App, "*app_instance", full_app, raising=False
        )
        notification = object()

        app._handle_notification_activation(notification)

        assert received == [notification]


class TestRequestedNotificationEvents:
    def test_catalog_covers_every_requested_event(self) -> None:
        assert set(app.NOTIFICATION_KEYS) == {
            "microphone_muted",
            "microphone_unmuted",
            "speakers_off",
            "speakers_on",
            "brightness_dimmed",
            "brightness_restored",
            "break_timer_started",
            "break_timer_finished",
            "data_range_changed",
            "mood_tip_generated",
            "mood_tip_pdf_exported",
            "csv_downloaded",
            "data_erased",
            "privacy_settings_changed",
            "sensitivity_changed",
            "ai_provider_changed",
            "app_paused",
            "app_resumed",
            "emotion_neutral",
            "emotion_happy",
            "emotion_surprised",
            "emotion_angry",
            "emotion_sad",
            "emotion_no_face",
            "emotion_multiple_faces",
            "license_activated",
            "license_deactivated",
            "app_quit",
        }

    def test_emotions_notify_once_per_transition(
        self, full_app, monkeypatch
    ) -> None:
        emotion_events = tuple(app.EMOTION_NOTIFICATION_EVENTS.values())
        calls = _capture_notifications(full_app, monkeypatch, *emotion_events)
        labels = ("neutral", "happy", "surprised", "angry", "sad", "no face")
        for label in labels:
            full_app._worker._set_result(EmotionResult(label, 0.8))
            full_app.refresh(None)
            full_app.refresh(None)
        assert [call[1] for call in calls] == [
            "Neutral detected",
            "Happy detected",
            "Surprised detected",
            "Angry detected",
            "Sad detected",
            "No face detected",
        ]

    def test_multiple_faces_reset_emotion_notification_transition(
        self, full_app, monkeypatch
    ) -> None:
        calls = _capture_notifications(full_app, monkeypatch, "emotion_happy")

        full_app._notify_emotion_transition("happy")
        full_app._notify_emotion_transition(app.MULTI_FACE_LABEL)
        full_app._notify_emotion_transition("happy")

        assert [call[1] for call in calls] == ["Happy detected", "Happy detected"]

    def test_multiple_faces_notify_once_per_transition_with_live_count(
        self, full_app, monkeypatch
    ) -> None:
        calls = _capture_notifications(
            full_app,
            monkeypatch,
            "emotion_multiple_faces",
        )

        full_app._notify_emotion_transition(app.MULTI_FACE_LABEL, face_count=3)
        full_app._notify_emotion_transition(app.MULTI_FACE_LABEL, face_count=4)
        full_app._notify_emotion_transition("happy")
        full_app._notify_emotion_transition(app.MULTI_FACE_LABEL, face_count=2)

        assert calls == [
            ("Moodito", "Multiple faces detected", "3 faces are currently visible."),
            ("Moodito", "Multiple faces detected", "2 faces are currently visible."),
        ]

    def test_pause_resume_sensitivity_and_ai_provider_notify(
        self, full_app, monkeypatch
    ) -> None:
        calls = _capture_notifications(
            full_app,
            monkeypatch,
            "app_paused",
            "app_resumed",
            "sensitivity_changed",
            "ai_provider_changed",
        )
        monkeypatch.setattr(app, "save_settings", lambda settings: None)

        full_app.toggle_pause(None)
        full_app.toggle_pause(None)
        full_app._apply_sensitivity_from_controls(
            {
                "happy": _FakeSegment(app.SENSITIVITY_LEVELS.index("high")),
                "angry": _FakeSegment(app.SENSITIVITY_LEVELS.index("low")),
            }
        )
        values = {"api_key": "k", "model": "gpt-4o"}
        full_app._apply_ai_provider("OpenAI", values)
        full_app._apply_ai_provider("OpenAI", values)

        assert [call[1] for call in calls] == [
            "Moodito paused",
            "Moodito resumed",
            "Sensitivity updated",
            "AI provider updated",
        ]
        assert "Happy" in calls[2][2] and "Angry" in calls[2][2]

    def test_data_range_notifies_only_when_changed(
        self, data_dir, monkeypatch
    ) -> None:
        _patch_window(monkeypatch, clicked=1, text="2026-06-21 22:10 to now")
        inst = _bare_app()
        calls = _capture_notifications(inst, monkeypatch, "data_range_changed")

        inst.set_stats_range(None)
        inst.set_stats_range(None)

        assert [call[1] for call in calls] == ["Data range changed"]
        assert "Jun 21, 2026 22:10" in calls[0][2]

    def test_mood_tip_notifies_only_for_success(
        self, full_app, monkeypatch
    ) -> None:
        calls = _capture_notifications(full_app, monkeypatch, "mood_tip_generated")
        handler = _FakeMoodHandler(full_app)

        full_app._finish_mood_report(handler, "A finished report")
        full_app._finish_mood_report(
            handler, f"{app.MOOD_TIP_ERROR_PREFIX}\noffline"
        )

        assert [call[1] for call in calls] == ["Mood Tip ready"]

    def test_pdf_notifies_only_after_successful_write(
        self, full_app, monkeypatch
    ) -> None:
        import AppKit

        class FakeSavePanel:
            @classmethod
            def savePanel(cls):
                return cls()

            def setNameFieldStringValue_(self, _value) -> None:
                pass

            def setAllowedContentTypes_(self, _value) -> None:
                pass

            def setAllowedFileTypes_(self, _value) -> None:
                pass

            def runModal(self):
                return AppKit.NSModalResponseOK

            def URL(self):
                return "report-url"

        fake_app = type(
            "FakeNSApp",
            (),
            {"activateIgnoringOtherApps_": lambda self, _active: None},
        )()
        monkeypatch.setattr(AppKit, "NSSavePanel", FakeSavePanel)
        monkeypatch.setattr(AppKit, "NSApp", fake_app)
        calls = _capture_notifications(
            full_app, monkeypatch, "mood_tip_pdf_exported"
        )
        handler = _FakeMoodHandler(full_app)
        handler.report_text = "A finished report"
        results = iter((True, False))
        monkeypatch.setattr(app, "write_report_pdf", lambda *args: next(results))

        full_app._save_mood_report_pdf(handler)
        full_app._save_mood_report_pdf(handler)

        assert [call[1] for call in calls] == ["Mood Tip PDF saved"]

    def test_csv_download_notifies_after_file_is_written(
        self, data_dir, monkeypatch
    ) -> None:
        monkeypatch.setenv("HOME", str(data_dir))
        (data_dir / "Downloads").mkdir()
        monkeypatch.setattr(app.subprocess, "run", lambda *args, **kwargs: None)
        inst = _bare_app()
        calls = _capture_notifications(inst, monkeypatch, "csv_downloaded")

        inst.export_csv(None)

        assert [call[1] for call in calls] == ["CSV downloaded"]
        assert "Downloads" in calls[0][2]

    def test_data_erase_cancel_is_quiet_and_success_notifies(
        self, data_dir, monkeypatch
    ) -> None:
        responses = iter((0, 1))
        monkeypatch.setattr(app.rumps, "alert", lambda *args, **kwargs: next(responses))
        monkeypatch.setattr(app, "save_stats", lambda *args, **kwargs: None)
        inst = _bare_app()
        calls = _capture_notifications(inst, monkeypatch, "data_erased")

        inst.reset_stats(None)
        inst.reset_stats(None)

        assert [call[1] for call in calls] == ["Data erased"]

    def test_license_events_are_delivered_on_main_thread(
        self, data_dir, monkeypatch
    ) -> None:
        inst = _bare_license_app(active=False)
        calls = _capture_notifications(
            inst, monkeypatch, "license_activated", "license_deactivated"
        )
        monkeypatch.setattr(app.rumps, "alert", lambda *args, **kwargs: None)
        monkeypatch.setattr(app, "license_instance_name", lambda: "Moodito on Mac")
        monkeypatch.setattr(
            app, "activate_license", lambda *args: (True, "activated", "iid")
        )

        inst._activate_worker("key")
        assert calls == []
        inst._consume_license_updates()
        monkeypatch.setattr(app, "deactivate_license", lambda *args: (True, "ok"))
        inst._deactivate_worker("key", "iid")
        assert len(calls) == 1
        inst._consume_license_updates()

        assert [call[1] for call in calls] == [
            "License activated",
            "License deactivated",
        ]

    def test_quit_notifies_after_state_is_flushed(self, monkeypatch) -> None:
        order = []
        monkeypatch.setattr(
            app, "save_stats", lambda *args, **kwargs: order.append("saved")
        )
        monkeypatch.setattr(
            app,
            "append_raw_samples",
            lambda rows: order.append("flushed") or True,
        )
        monkeypatch.setattr(
            app.rumps, "quit_application", lambda: order.append("quit")
        )
        inst = _bare_app()
        inst._worker = type("Worker", (), {"stop": lambda self: order.append("stopped")})()
        calls = _capture_notifications(inst, monkeypatch, "app_quit")

        inst.quit_app(None)

        assert [call[1] for call in calls] == ["Moodito quit"]
        assert order == ["saved", "flushed", "stopped", "quit"]


