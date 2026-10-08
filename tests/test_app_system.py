"""Things an installed app depends on: per-system folders, and the built-in self-test."""

from pathlib import Path

from strainlink.app import selftest, system


def test_user_data_folder_per_system(tmp_path):
    home = tmp_path / "home"
    assert system.data_dir_for("darwin", "posix", {}, home) == home / "Library" / "Application Support" / "StrainLink"
    assert system.data_dir_for("win32", "nt", {"APPDATA": str(tmp_path / "Roaming")}, home) == tmp_path / "Roaming" / "StrainLink"
    assert system.data_dir_for("win32", "nt", {}, home) == home / "AppData" / "Roaming" / "StrainLink"
    assert system.data_dir_for("linux", "posix", {}, home) == home / ".local" / "share" / "StrainLink"
    assert system.user_data_dir().name == "StrainLink"


def test_running_from_source_is_not_packaged():
    assert system.is_packaged() is False


def test_self_test_passes_and_writes_its_log(monkeypatch, tmp_path):
    monkeypatch.setattr(selftest, "user_data_dir", lambda: tmp_path)
    assert selftest.run() == 0
    log = (tmp_path / selftest.LOG_NAME).read_text(encoding="utf-8")
    assert "SELF-TEST OK" in log and "known-answer check" in log
