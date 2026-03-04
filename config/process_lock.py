"""
Process lock to prevent multiple instances of the application from running
simultaneously. This is critical because the app loads large ML models into
GPU VRAM, and duplicate instances would cause excessive memory usage.
"""

import os


class ProcessLock:
    """File-based process lock using PID files.

    Provides a re-entrant lock mechanism that detects and overrides stale locks
    left behind by crashed processes.

    Args:
        lock_path: Filesystem path where the lock file will be created.
    """

    def __init__(self, lock_path: str) -> None:
        self.lock_path = lock_path

    def _read_lock_pid(self) -> int | None:
        """Read the PID stored in the lock file.

        Returns:
            The PID as an integer, or None if the file doesn't exist or
            contains invalid content.
        """
        try:
            with open(self.lock_path, "r") as f:
                content = f.read().strip()
            return int(content)
        except (FileNotFoundError, ValueError):
            return None

    def _write_lock_pid(self, pid: int) -> None:
        """Write a PID to the lock file, creating parent directories if needed."""
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        with open(self.lock_path, "w") as f:
            f.write(str(pid))

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check whether a process with the given PID is currently running.

        Uses ``os.kill`` with signal 0, which performs the error-checking
        step of sending a signal without actually delivering one.

        Returns:
            True if the process exists, False otherwise.

        Notes:
            - If os.kill succeeds, the process exists and we can signal it.
            - If it raises PermissionError (EPERM), the process exists but we
              lack permission to signal it -- it is still running.
            - If it raises ProcessLookupError (ESRCH), the process does not exist.
        """
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            # EPERM: process exists but we are not allowed to signal it.
            return True
        except ProcessLookupError:
            # ESRCH: no such process.
            return False
        except OSError:
            # Catch-all for any other OS-level error -- treat as not running.
            return False

    def acquire(self) -> bool:
        """Attempt to acquire the process lock.

        Behaviour:
        - If the lock file does not exist, create it with the current PID and
          return True.
        - If the lock file exists and contains the current PID (re-entrant),
          return True.
        - If the lock file exists but the recorded PID is no longer running
          (stale lock), override it with the current PID and return True.
        - If the lock file exists and the recorded PID is still running,
          return False.

        Returns:
            True if the lock was successfully acquired, False otherwise.
        """
        my_pid = os.getpid()
        existing_pid = self._read_lock_pid()

        if existing_pid is None:
            # No lock file or unreadable content -- claim it.
            self._write_lock_pid(my_pid)
            return True

        if existing_pid == my_pid:
            # Re-entrant: we already own the lock.
            return True

        if not self._is_process_running(existing_pid):
            # Stale lock from a dead process -- override.
            self._write_lock_pid(my_pid)
            return True

        # Another live process holds the lock.
        return False

    def release(self) -> None:
        """Release the process lock.

        The lock file is removed **only** if it currently contains this
        process's PID, preventing one instance from accidentally releasing
        another's lock.
        """
        my_pid = os.getpid()
        existing_pid = self._read_lock_pid()

        if existing_pid == my_pid:
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass

    def is_locked_by_another(self) -> bool:
        """Check whether another running process holds the lock.

        Returns:
            True if the lock file exists, contains a PID that is still
            running, and that PID is not the current process. False otherwise.
        """
        existing_pid = self._read_lock_pid()

        if existing_pid is None:
            return False

        if existing_pid == os.getpid():
            return False

        return self._is_process_running(existing_pid)
