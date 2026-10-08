"""Тесты персистентного хранилища user_id (с временной директорией)."""
import json

import pytest

import user_id_store


@pytest.fixture(autouse=True)
def _patch_user_file(tmp_path, monkeypatch):
    """Перенаправляем файл хранилища во временную директорию."""
    user_file = tmp_path / "user_id.json"
    monkeypatch.setattr(user_id_store, "_USER_FILE", user_file)
    yield user_file


def test_load_missing_returns_none(_patch_user_file):
    assert user_id_store.load_user_id() is None


def test_save_and_load_roundtrip(_patch_user_file):
    user_id_store.save_user_id("user-123")
    assert user_id_store.load_user_id() == "user-123"
    saved = json.loads(_patch_user_file.read_text())
    assert saved == {"user_id": "user-123"}


def test_clear_removes_file(_patch_user_file):
    user_id_store.save_user_id("user-123")
    user_id_store.clear_user_id()
    assert not _patch_user_file.exists()
    assert user_id_store.load_user_id() is None


def test_load_corrupted_file_returns_none(_patch_user_file):
    _patch_user_file.write_text("{not valid json")
    assert user_id_store.load_user_id() is None


def test_load_file_without_user_id_key_returns_none(_patch_user_file):
    _patch_user_file.write_text(json.dumps({"other": "value"}))
    assert user_id_store.load_user_id() is None
