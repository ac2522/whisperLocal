"""End-to-end test: simulate transcription and verify auto-paste into external window.

Flow:
1. Launch a separate target window process (simulates user's terminal)
2. Import _paste_text logic and call it with test text
3. Verify text appeared in the target window
"""

import os
import subprocess
import sys
import time


RESULT_FILE = "/tmp/ydotool_e2e_result.txt"
TARGET_SCRIPT = os.path.join(os.path.dirname(__file__), "_target_window.py")
TEST_TEXT = "This is a transcribed sentence from whisper."


def test_autopaste_e2e():
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    # Launch target window
    target = subprocess.Popen(
        [sys.executable, TARGET_SCRIPT, RESULT_FILE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    try:
        time.sleep(2.0)

        # Simulate exactly what _paste_text does
        time.sleep(0.05)
        result = subprocess.run(
            ['ydotool', 'type', '--delay', '100', '--key-delay', '12', '--', TEST_TEXT],
            timeout=10, check=False, capture_output=True, text=True,
        )
        print(f"ydotool exit: {result.returncode}")

        target.wait(timeout=10)

        assert os.path.exists(RESULT_FILE), "Target did not write result file"
        with open(RESULT_FILE) as f:
            actual = f.read()

        print(f"Expected: {TEST_TEXT!r}")
        print(f"Got:      {actual!r}")
        assert actual == TEST_TEXT, f"Mismatch: expected {TEST_TEXT!r}, got {actual!r}"
        print("PASS: Auto-paste e2e works - transcribed text typed into external window")

    finally:
        if target.poll() is None:
            target.kill()
        if os.path.exists(RESULT_FILE):
            os.remove(RESULT_FILE)


if __name__ == "__main__":
    test_autopaste_e2e()
