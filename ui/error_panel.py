"""Collapsible error/log panel widget for displaying in-app log entries."""

import logging

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
)
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtCore import Qt


class QtLogHandler(logging.Handler):
    """Logging handler that forwards log records to an ErrorPanel widget."""

    def __init__(self, panel: "ErrorPanel"):
        super().__init__()
        self.panel = panel

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.panel.append_log(msg, record.levelno)
        except Exception:
            self.handleError(record)


class ErrorPanel(QWidget):
    """Collapsible panel that displays application log entries.

    The panel installs a logging handler on the root logger so that any
    ``logger.info()`` / ``logger.error()`` call from *any* module is
    automatically captured and displayed inside the panel's text area.
    """

    MAX_ENTRIES = 50

    # Colour map: logging level -> HTML hex colour
    _LEVEL_COLOURS = {
        logging.ERROR: "#cc0000",
        logging.WARNING: "#cc7700",
        logging.INFO: "#333333",
        logging.DEBUG: "#888888",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._entry_count = 0

        # --- Header bar ---------------------------------------------------
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self._toggle_button = QPushButton("Show Logs")
        self._toggle_button.clicked.connect(self._toggle)
        header_layout.addWidget(self._toggle_button)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self._status_label, stretch=1)

        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self._clear)
        header_layout.addWidget(self._clear_button)

        # --- Text area (hidden by default) ---------------------------------
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setMaximumHeight(200)
        self._text_edit.setVisible(False)

        mono_font = QFont("monospace", 9)
        mono_font.setStyleHint(QFont.Monospace)
        self._text_edit.setFont(mono_font)

        # --- Main layout ---------------------------------------------------
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header_layout)
        layout.addWidget(self._text_edit)
        self.setLayout(layout)

        # --- Install logging handler ---------------------------------------
        self._install_log_handler()

    # ------------------------------------------------------------------
    # Logging integration
    # ------------------------------------------------------------------

    def _install_log_handler(self) -> None:
        """Create a :class:`QtLogHandler` and attach it to the root logger."""
        handler = QtLogHandler(self)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)

        logging.getLogger().addHandler(handler)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_log(self, message: str, level: int) -> None:
        """Append a coloured log *message* to the text area.

        Parameters
        ----------
        message:
            Pre-formatted log string.
        level:
            The numeric logging level (e.g. ``logging.ERROR``).
        """
        colour = self._LEVEL_COLOURS.get(level, self._LEVEL_COLOURS[logging.INFO])
        html = f'<span style="color:{colour};">{message}</span>'
        self._text_edit.append(html)
        self._entry_count += 1

        # Auto-expand on ERROR
        if level >= logging.ERROR and not self._text_edit.isVisible():
            self._toggle()

        # Trim old entries when over the limit
        if self._entry_count > self.MAX_ENTRIES:
            self._trim_entries()

        # Auto-scroll to bottom
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        """Show or hide the text area and update the toggle button label."""
        visible = not self._text_edit.isVisible()
        self._text_edit.setVisible(visible)
        self._toggle_button.setText("Hide Logs" if visible else "Show Logs")

    def _clear(self) -> None:
        """Clear all log entries from the text area."""
        self._text_edit.clear()
        self._entry_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trim_entries(self) -> None:
        """Remove the oldest entries so that at most MAX_ENTRIES remain."""
        doc = self._text_edit.document()
        while doc.blockCount() > self.MAX_ENTRIES and doc.blockCount() > 1:
            cursor = QTextCursor(doc.begin())
            cursor.select(QTextCursor.BlockUnderCursor)
            # Also grab the trailing newline so we don't accumulate blanks
            cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self._entry_count -= 1
