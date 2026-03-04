#!/usr/bin/env python3

import os
os.environ['ALSA_DEBUG'] = '0'

import sys
import signal
from pathlib import Path

from config.logging_setup import setup_logging
from config.process_lock import ProcessLock

SETTINGS_DIR = os.path.join(Path.home(), '.whisper2text')
LOCK_FILE = os.path.join(SETTINGS_DIR, 'app.lock')


def main():
    setup_logging()

    lock = ProcessLock(LOCK_FILE)
    if not lock.acquire():
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.warning(None, "Already Running",
                          "Another instance of Speech to Text is already running.")
        sys.exit(1)

    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QIcon
        from config.paths import ICON_NORMAL
        from ui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon(ICON_NORMAL))

        window = MainWindow()
        window.show()

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(app.exec_())
    finally:
        lock.release()


if __name__ == '__main__':
    main()
