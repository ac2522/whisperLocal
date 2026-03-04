"""Tests for config.process_lock.ProcessLock."""

import os
import tempfile

import pytest

from config.process_lock import ProcessLock


@pytest.fixture
def lock_path(tmp_path):
    """Return a temporary lock file path (file does not exist yet)."""
    return str(tmp_path / "test.lock")


@pytest.fixture
def lock(lock_path):
    """Return a ProcessLock instance with automatic cleanup."""
    pl = ProcessLock(lock_path)
    yield pl
    # Ensure the lock file is cleaned up after each test.
    if os.path.exists(lock_path):
        os.remove(lock_path)


# ---- acquire / release basics ------------------------------------------------


class TestAcquireRelease:
    def test_acquire_creates_lock_file(self, lock, lock_path):
        """acquire() should create the lock file and return True."""
        assert lock.acquire() is True
        assert os.path.exists(lock_path)

    def test_lock_file_contains_current_pid(self, lock, lock_path):
        """The lock file should contain the current process PID."""
        lock.acquire()
        with open(lock_path) as f:
            assert int(f.read().strip()) == os.getpid()

    def test_release_removes_lock_file(self, lock, lock_path):
        """release() should remove the lock file."""
        lock.acquire()
        lock.release()
        assert not os.path.exists(lock_path)

    def test_release_without_acquire_is_noop(self, lock, lock_path):
        """release() on a non-existent lock should not raise."""
        lock.release()  # should not raise
        assert not os.path.exists(lock_path)

    def test_release_does_not_remove_other_pid_lock(self, lock, lock_path):
        """release() must not delete a lock file owned by another PID."""
        # Write a foreign PID into the lock file.
        with open(lock_path, "w") as f:
            f.write("999999999")
        lock.release()
        # The file should still be there because it belongs to PID 999999999.
        assert os.path.exists(lock_path)


# ---- re-entrant acquire ------------------------------------------------------


class TestReentrant:
    def test_double_acquire_same_process(self, lock):
        """Calling acquire() twice from the same process should succeed both times."""
        assert lock.acquire() is True
        assert lock.acquire() is True

    def test_reentrant_acquire_preserves_pid(self, lock, lock_path):
        """A second acquire() should not change the recorded PID."""
        lock.acquire()
        lock.acquire()
        with open(lock_path) as f:
            assert int(f.read().strip()) == os.getpid()


# ---- stale lock override -----------------------------------------------------


class TestStaleLock:
    def _write_stale_lock(self, lock_path: str) -> int:
        """Write a lock file with a PID that is guaranteed not to be running.

        We pick a very high PID that is almost certainly unused. As extra
        safety we verify it is indeed not running before returning.
        """
        stale_pid = 4_000_000  # Very unlikely to be a real running process.
        # Walk upward if, against all odds, the PID is alive.
        while True:
            try:
                os.kill(stale_pid, 0)
                stale_pid += 1  # pragma: no cover
            except (OSError, ProcessLookupError):
                break
        with open(lock_path, "w") as f:
            f.write(str(stale_pid))
        return stale_pid

    def test_stale_lock_is_overridden(self, lock, lock_path):
        """acquire() should override a lock file left by a dead process."""
        self._write_stale_lock(lock_path)
        assert lock.acquire() is True
        with open(lock_path) as f:
            assert int(f.read().strip()) == os.getpid()

    def test_stale_lock_is_not_locked_by_another(self, lock, lock_path):
        """is_locked_by_another() should return False for a stale lock."""
        self._write_stale_lock(lock_path)
        assert lock.is_locked_by_another() is False


# ---- is_locked_by_another ----------------------------------------------------


class TestIsLockedByAnother:
    def test_no_lock_file(self, lock):
        """Should return False when no lock file exists."""
        assert lock.is_locked_by_another() is False

    def test_locked_by_self(self, lock):
        """Should return False when the lock is held by the current process."""
        lock.acquire()
        assert lock.is_locked_by_another() is False

    def test_locked_by_running_process(self, lock, lock_path):
        """Should return True when the lock is held by another running process.

        We use PID 1 (init/systemd) which is always running on Linux.
        """
        with open(lock_path, "w") as f:
            f.write("1")
        assert lock.is_locked_by_another() is True

    def test_locked_by_dead_process(self, lock, lock_path):
        """Should return False when the lock is held by a dead process."""
        dead_pid = 4_000_000
        while True:
            try:
                os.kill(dead_pid, 0)
                dead_pid += 1  # pragma: no cover
            except (OSError, ProcessLookupError):
                break
        with open(lock_path, "w") as f:
            f.write(str(dead_pid))
        assert lock.is_locked_by_another() is False


# ---- acquire blocked by another process --------------------------------------


class TestAcquireBlocked:
    def test_acquire_fails_when_another_process_holds_lock(self, lock, lock_path):
        """acquire() should return False when another running process holds the lock.

        We use PID 1 (init/systemd) which is always running on Linux.
        """
        with open(lock_path, "w") as f:
            f.write("1")
        assert lock.acquire() is False

    def test_lock_file_unchanged_after_failed_acquire(self, lock, lock_path):
        """A failed acquire() must not alter the existing lock file."""
        with open(lock_path, "w") as f:
            f.write("1")
        lock.acquire()
        with open(lock_path) as f:
            assert f.read().strip() == "1"
