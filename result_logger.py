"""
result_logger.py

Xử lý lưu kết quả sau mỗi QR rời khung hình:
  - OK  → ghi 1 dòng vào CSV  (timestamp, text, track_id, duration)
  - NG  → lưu ảnh frame cuối có vẽ ROI đỏ quanh QR bị NG

Thiết kế:
  - ResultLogger nhận frame cuối của track qua set_last_frame().
    GUI gọi hàm này mỗi khi detect xong, truyền frame gốc (chưa vẽ).
  - Khi on_finalize được gọi bởi QRTracker, gọi log(track, frame).
  - Thread-safe: ghi file trong thread detect, không block UI.
  - Tạo thư mục tự động nếu chưa có.
  - CSV có header; nếu file đã tồn tại thì append (không ghi header lại).
"""

import os
import csv
import threading
import time
import logging
from typing import Optional, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Màu ROI vẽ lên ảnh NG (BGR)
NG_BOX_COLOR  = (0, 0, 220)
NG_BOX_THICK  = 3
NG_LABEL_BG   = (0, 0, 170)
NG_TEXT_COLOR = (255, 255, 255)

CSV_HEADER = ["timestamp", "datetime", "track_id", "qr_text", "duration_s"]


class ResultLogger:
    def __init__(self, output_dir: str = "results"):
        self._dir = output_dir
        self._lock = threading.Lock()
        # frame cuối của mỗi track: {track_id: (frame_bgr, det_dict)}
        self._last_frames: Dict[int, tuple] = {}
        self._ensure_dirs()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_output_dir(self, output_dir: str):
        """Đổi thư mục lưu (gọi từ UI khi user thay đổi trong Settings)."""
        with self._lock:
            self._dir = output_dir
            self._ensure_dirs()

    def update_last_frame(self, track_id: int, frame: np.ndarray, det: Dict[str, Any]):
        """
        Lưu frame + detection box của track để dùng khi finalize.
        Gọi từ detection loop SAU khi tracker.update() đã gán track_id.
        Chỉ lưu frame của track đang active, xoá track đã done tự động.
        """
        with self._lock:
            # Clone frame để tránh race với detection thread đang mutate
            self._last_frames[track_id] = (frame.copy(), dict(det))

    def log(self, track) -> bool:
        """
        Gọi từ on_finalize callback.
        track.status == "OK" → ghi CSV
        track.status == "NG" → lưu ảnh
        Trả về True nếu ghi thành công.
        """
        try:
            with self._lock:
                frame_data = self._last_frames.pop(track.id, None)

            if track.status == "OK":
                return self._write_csv(track)
            else:
                return self._save_ng_image(track, frame_data)
        except Exception as e:
            logger.error(f"ResultLogger.log error: {e}")
            return False

    def cleanup_track(self, track_id: int):
        """Xoá frame cache của track đã xử lý xong (gọi khi track bị drop im lặng)."""
        with self._lock:
            self._last_frames.pop(track_id, None)

    @property
    def output_dir(self) -> str:
        return self._dir

    @property
    def ok_dir(self) -> str:
        return self._dir   # CSV nằm thẳng trong output_dir

    @property
    def ng_dir(self) -> str:
        return os.path.join(self._dir, "ng_images")

    @property
    def csv_path(self) -> str:
        return os.path.join(self._dir, "ok_results.csv")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_dirs(self):
        os.makedirs(self._dir, exist_ok=True)
        os.makedirs(self.ng_dir, exist_ok=True)

    def _write_csv(self, track) -> bool:
        path = self.csv_path
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(CSV_HEADER)
                ts = time.time()
                dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                duration = round(ts - track.first_seen, 2)
                writer.writerow([
                    f"{ts:.3f}",
                    dt,
                    track.id,
                    track.best_text,
                    duration,
                ])
            logger.info(f"[OK] CSV logged: {track.best_text!r}")
            return True
        except Exception as e:
            logger.error(f"CSV write error: {e}")
            return False

    def _save_ng_image(self, track, frame_data) -> bool:
        if frame_data is None:
            logger.warning(f"[NG] No frame cached for track {track.id}, skipping image save.")
            return False

        frame_bgr, det = frame_data
        img = frame_bgr.copy()

        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        # Vẽ ROI đỏ đậm quanh QR bị NG
        cv2.rectangle(img, (x1, y1), (x2, y2), NG_BOX_COLOR, NG_BOX_THICK)

        # Label "NG" với nền đỏ
        label = "NG"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), _ = cv2.getTextSize(label, font, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - lh - 14), (x1 + lw + 12, y1), NG_LABEL_BG, -1)
        cv2.putText(img, label, (x1 + 6, y1 - 6), font, 0.9, NG_TEXT_COLOR, 2)

        # Timestamp nhỏ ở góc trên trái ảnh
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, ts_str, (10, 24), font, 0.55, (200, 200, 200), 1)

        # Tên file: ng_YYYYMMDD_HHMMSS_mmm_trackID.jpg
        ts_file = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"ng_{ts_file}_{ms:03d}_t{track.id}.jpg"
        filepath = os.path.join(self.ng_dir, filename)

        ok = cv2.imwrite(filepath, img)
        if ok:
            logger.info(f"[NG] Image saved: {filepath}")
        else:
            logger.error(f"[NG] imwrite failed: {filepath}")
        return ok