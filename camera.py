import cv2
import threading
import time
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages camera capture with auto-reconnect support.
    Designed for Dino-Lite USB cameras but works with any OpenCV-compatible camera.
    """

    def __init__(self, camera_index: int = 0, on_status_change: Optional[Callable] = None):
        self.camera_index = camera_index
        self.on_status_change = on_status_change

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_delay = 2.0  # seconds

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Attempt to connect to the camera."""
        try:
            if self._cap:
                self._cap.release()

            self._cap = cv2.VideoCapture(self.camera_index)
            if self._cap.isOpened():
                # Giảm buffer nội bộ của driver để tránh đọc phải frame "quá khứ"
                # (giảm độ trễ giữa lúc camera thấy ảnh và lúc app đọc được ảnh đó)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # MJPG thường cho fps cao hơn / băng thông thấp hơn so với
                # định dạng mặc định trên nhiều webcam USB (bao gồm Dino-Lite)
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                # Optimal settings for Dino-Lite
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                self._cap.set(cv2.CAP_PROP_FPS, 30)
                self._connected = True
                logger.info(f"Camera {self.camera_index} connected.")
                self._notify_status("Connected")
                return True
            else:
                self._connected = False
                self._notify_status("Disconnected")
                return False
        except Exception as e:
            logger.error(f"Camera connect error: {e}")
            self._connected = False
            self._notify_status("Error")
            return False

    def disconnect(self):
        """Disconnect from camera."""
        self._connected = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._notify_status("Disconnected")
        logger.info("Camera disconnected.")

    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera capture thread started.")

    def stop(self):
        """Stop the capture thread and disconnect."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.disconnect()
        logger.info("Camera stopped.")

    def read(self) -> Optional[object]:
        """Return the latest captured frame (thread-safe)."""
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def _capture_loop(self):
        """Main capture loop with auto-reconnect logic."""
        while self._running:
            if not self._connected:
                logger.info("Attempting to reconnect camera...")
                self._notify_status("Reconnecting...")
                if not self.connect():
                    time.sleep(self._reconnect_delay)
                    continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._frame = frame
            else:
                # Frame read failed – camera likely disconnected
                logger.warning("Frame read failed. Triggering reconnect.")
                self._connected = False
                self._notify_status("Disconnected")
                time.sleep(0.5)

    def _notify_status(self, status: str):
        if self.on_status_change:
            try:
                self.on_status_change(status)
            except Exception:
                pass