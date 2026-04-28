"""API key storage for cloud transcription providers.

Resolution order: DEEPGRAM_API_KEY env var, then OS keyring entry
(service "whisperLocal", username "deepgram"). The `keyring` module
is imported lazily inside each function so a missing or broken
backend never blocks app startup or local-engine use.
"""

import logging
import os

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "whisperLocal"
DEEPGRAM_USERNAME = "deepgram"


def get_deepgram_key() -> str | None:
    """Return the configured Deepgram API key, or None.

    Env var wins over the keyring so users can override per shell or
    per systemd unit without touching stored secrets.
    """
    env = os.environ.get("DEEPGRAM_API_KEY")
    if env:
        return env
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, DEEPGRAM_USERNAME)
    except Exception:
        logger.warning("keyring unavailable", exc_info=True)
        return None


def set_deepgram_key(value: str) -> None:
    """Persist *value* into the OS keyring."""
    import keyring
    keyring.set_password(KEYRING_SERVICE, DEEPGRAM_USERNAME, value)


def clear_deepgram_key() -> None:
    """Remove the stored keyring entry. No-op if there is none."""
    import keyring
    try:
        keyring.delete_password(KEYRING_SERVICE, DEEPGRAM_USERNAME)
    except keyring.errors.PasswordDeleteError:
        pass


def has_deepgram_key() -> bool:
    """Return True if a key is available from any source."""
    return get_deepgram_key() is not None


def get_key_source() -> str | None:
    """Return 'env', 'keyring', or None — for the Settings status label."""
    if os.environ.get("DEEPGRAM_API_KEY"):
        return "env"
    try:
        import keyring
        if keyring.get_password(KEYRING_SERVICE, DEEPGRAM_USERNAME):
            return "keyring"
    except Exception:
        pass
    return None
