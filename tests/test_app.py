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
    inst._stats_range_start = app.datetime(2026, 6, 21, 22, 0, 0)
    inst._stats_range_end = None
    inst._stats_live_24h = False
    inst._settings = {}
    inst._range_stats = {k: {"seconds": 0.0, "count": 0} for k in app.STAT_KEYS}
    inst._range_last_state = None
    inst._hourly_activity = [0.0] * 24
    inst._stats_items = {k: rumps.MenuItem(k) for k in app.STAT_KEYS}
    inst._stats_header_item = rumps.MenuItem("header")
    inst._stats_total_item = rumps.MenuItem("total")
    inst._stats_since_item = rumps.MenuItem("since")
    inst._stats_range_item = rumps.MenuItem("range", callback=inst.set_stats_range)
    inst._stats_live_item = rumps.MenuItem("live")
    inst._stats_activity_header_item = rumps.MenuItem("activity")
    inst._stats_activity_item = rumps.MenuItem("spark")
    inst._stats_activity_axis_item = rumps.MenuItem("axis")
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
            app, "append_raw_samples", lambda rows: flushed.extend(rows)
        )
        inst = _bare_app()
        ticks = int(10 / app.UI_REFRESH_INTERVAL) + 1
        for _ in range(ticks):
            inst._accumulate_stats("happy", 0.5)
        assert flushed  # raw samples were flushed
        assert inst._raw_buffer == []  # buffer cleared after flush


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
    inst._license_dirty = app.threading.Event()
    inst._license_status_item = rumps.MenuItem("status")
    inst._license_activate_item = rumps.MenuItem("activate")
    inst._license_deactivate_item = rumps.MenuItem("deactivate")
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

    def test_visibility_when_inactive(self) -> None:
        inst = _bare_license_app(active=False)
        inst._apply_license_visibility()
        assert inst._license_activate_item.hidden is False
        assert inst._license_deactivate_item.hidden is True
        assert inst._bmc_menu.hidden is False
        assert "Not licensed" in inst._license_status_item.title

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
    return app.MooditoApp()


class TestMooditoAppInit:
    def test_builds_menu_and_records_start_time(self, full_app) -> None:
        assert full_app._stats_started_at is not None
        assert full_app._raw_buffer == []
        assert "Download (csv)" in full_app._stats_export_item.title
        assert set(full_app._stats_items) == set(app.STAT_KEYS)

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
    def test_emojis_and_labels_show_emoji_and_text(self, full_app) -> None:
        full_app._show_emojis = True
        full_app._show_labels = True
        full_app._last_render = None
        full_app._render_emotion(EmotionResult("happy", 0.8))
        assert full_app.title == "😀 happy"
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


