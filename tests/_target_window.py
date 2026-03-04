"""Helper: a standalone window that receives typed text, writes result to file."""
import sys
from PyQt5.QtWidgets import QApplication, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

output_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ydotool_target_result.txt"

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("PASTE_TARGET")
window.resize(400, 200)
layout = QVBoxLayout(window)
text_edit = QTextEdit()
layout.addWidget(text_edit)
window.show()
window.activateWindow()
window.raise_()
text_edit.setFocus()

def save_and_quit():
    with open(output_file, "w") as f:
        f.write(text_edit.toPlainText())
    app.quit()

# Auto-quit after 6 seconds
QTimer.singleShot(6000, save_and_quit)
app.exec_()
