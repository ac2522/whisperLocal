"""PyInstaller runtime hook: add pywhispercpp.libs/ to LD_LIBRARY_PATH.

When running as a frozen PyInstaller bundle, the CUDA shared libraries
shipped inside pywhispercpp.libs/ must be discoverable by the dynamic
linker at runtime.
"""

import os
import sys

if getattr(sys, 'frozen', False):
    libs_dir = os.path.join(sys._MEIPASS, 'pywhispercpp.libs')
    if os.path.isdir(libs_dir):
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if libs_dir not in ld_path:
            os.environ['LD_LIBRARY_PATH'] = libs_dir + ':' + ld_path if ld_path else libs_dir
