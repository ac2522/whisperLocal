"""Tests for config.api_keys."""

import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure DEEPGRAM_API_KEY does not leak from the developer's shell."""
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    yield


@pytest.fixture
def fake_keyring(monkeypatch):
    """Patch the imported `keyring` module inside config.api_keys.

    The real module is imported lazily *inside each function*, so we
    monkeypatch sys.modules so the lazy import resolves to our fake.
    """
    fake = MagicMock()
    fake._store = {}

    def _set(service, user, value):
        fake._store[(service, user)] = value
    def _get(service, user):
        return fake._store.get((service, user))
    def _del(service, user):
        if (service, user) not in fake._store:
            from keyring.errors import PasswordDeleteError
            raise PasswordDeleteError("not found")
        del fake._store[(service, user)]

    fake.set_password = _set
    fake.get_password = _get
    fake.delete_password = _del

    import keyring as _real
    fake.errors = _real.errors

    monkeypatch.setitem(__import__("sys").modules, "keyring", fake)
    yield fake


class TestGetKey:
    def test_returns_env_when_set(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "env-value")
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() == "env-value"

    def test_returns_keyring_when_env_missing(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "ring-value"
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() == "ring-value"

    def test_env_wins_over_keyring(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "env-value")
        fake_keyring._store[("whisperLocal", "deepgram")] = "ring-value"
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() == "env-value"

    def test_returns_none_when_neither_present(self, fake_keyring):
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() is None

    def test_returns_none_when_keyring_raises(self, monkeypatch):
        # Simulate a keyring backend that blows up on any access.
        fake = MagicMock()
        fake.get_password.side_effect = RuntimeError("no backend")
        monkeypatch.setitem(__import__("sys").modules, "keyring", fake)
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() is None


class TestSetClear:
    def test_set_writes_to_keyring(self, fake_keyring):
        from config.api_keys import set_deepgram_key
        set_deepgram_key("new-value")
        assert fake_keyring._store[("whisperLocal", "deepgram")] == "new-value"

    def test_clear_removes_from_keyring(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "old"
        from config.api_keys import clear_deepgram_key
        clear_deepgram_key()
        assert ("whisperLocal", "deepgram") not in fake_keyring._store

    def test_clear_is_idempotent(self, fake_keyring):
        from config.api_keys import clear_deepgram_key
        clear_deepgram_key()  # no entry yet — must not raise


class TestHasKey:
    def test_true_when_env_set(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
        from config.api_keys import has_deepgram_key
        assert has_deepgram_key() is True

    def test_true_when_keyring_set(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "x"
        from config.api_keys import has_deepgram_key
        assert has_deepgram_key() is True

    def test_false_when_neither(self, fake_keyring):
        from config.api_keys import has_deepgram_key
        assert has_deepgram_key() is False


class TestGetKeySource:
    def test_returns_env_when_env_set(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
        from config.api_keys import get_key_source
        assert get_key_source() == "env"

    def test_returns_keyring_when_only_keyring_set(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "x"
        from config.api_keys import get_key_source
        assert get_key_source() == "keyring"

    def test_env_takes_precedence(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
        fake_keyring._store[("whisperLocal", "deepgram")] = "y"
        from config.api_keys import get_key_source
        assert get_key_source() == "env"

    def test_returns_none_when_nothing_set(self, fake_keyring):
        from config.api_keys import get_key_source
        assert get_key_source() is None
