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
