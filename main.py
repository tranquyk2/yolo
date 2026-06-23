"""
QR Detection System – V1
Entry point: python main.py
"""
import logging
import sys
import os


# Fix: khi chạy ở chế độ windowed (--windowed trong PyInstaller, hoặc pythonw),
# sys.stdout/sys.stderr = None khiến các thư viện log (ultralytics, tqdm...) bị crash
# với lỗi "'NoneType' object has no attribute 'write'"
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# Ensure the project directory is always first in sys.path
# so local modules (detector.py, camera.py, etc.) take priority
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import MainApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def main():
    app = MainApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()