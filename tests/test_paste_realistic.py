"""Integration test: ydotool types into a SEPARATE process window.

Simulates the real scenario:
1. Launch a separate Python process with a Qt text editor (simulates terminal/browser)
2. Wait for it to get focus
3. Call ydotool type from THIS process (simulates our app's background thread)
4. The target process saves what it received to a file
5. We read the file and verify
"""

import os
import subprocess
import sys
import time


RESULT_FILE = "/tmp/ydotool_target_result.txt"
TARGET_SCRIPT = os.path.join(os.path.dirname(__file__), "_target_window.py")
TEST_STRING = "hello from separate process"


def test_ydotool_types_into_separate_process():
    # Clean up
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    # Launch target window in a separate process
    target_proc = subprocess.Popen(
        [sys.executable, TARGET_SCRIPT, RESULT_FILE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        # Wait for the target window to appear and get focus
        time.sleep(2.0)

        # Type into it from THIS process using ydotool
        result = subprocess.run(
            ["ydotool", "type", "--delay", "100", "--key-delay", "12", "--", TEST_STRING],
            capture_output=True, text=True, timeout=10,
        )
        print(f"ydotool exit: {result.returncode}, stderr: {result.stderr.strip()}")

        # Wait for target to save and exit (it auto-quits after 6s)
        target_proc.wait(timeout=10)
        target_stderr = target_proc.stderr.read().decode()
        if target_stderr:
            print(f"Target stderr: {target_stderr.strip()}")

        # Read the result
        assert os.path.exists(RESULT_FILE), "Target window did not write result file"
        with open(RESULT_FILE) as f:
            actual = f.read()

        print(f"Expected: {TEST_STRING!r}")
        print(f"Got:      {actual!r}")
        assert actual == TEST_STRING, f"Expected {TEST_STRING!r}, got {actual!r}"
        print("PASS: ydotool typed into separate process window successfully")

    finally:
        if target_proc.poll() is None:
            target_proc.kill()
        if os.path.exists(RESULT_FILE):
            os.remove(RESULT_FILE)


if __name__ == "__main__":
    test_ydotool_types_into_separate_process()
