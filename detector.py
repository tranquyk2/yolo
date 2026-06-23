import os
import logging
from typing import List, Dict, Any

import numpy as np
import cv2
import zxingcpp

logger = logging.getLogger(__name__)

# Kích thước tối thiểu cạnh ngắn của crop (px) trước khi gửi vào zxing-cpp.
# Nếu crop nhỏ hơn thì upscale lên để tăng tỉ lệ decode thành công.
MIN_CROP_SHORT_SIDE = 120


class QRDetector:
    """
    YOLO có trách nhiệm TÌM vị trí QR (location).
    zxing-cpp có trách nhiệm ĐỌC (decode) nội dung QR trong vùng đó.
    Kết quả cuối cùng (OK/NG) dựa vào có giải mã được hay không,
    không chỉ dựa vào confidence của YOLO.

    Thay đổi so với v1:
    - Upscale crop lên ít nhất MIN_CROP_SHORT_SIDE px trước khi decode:
      giảm miss-decode do QR quá nhỏ trong frame (băng chuyền xa camera).
    - Thêm confirmed_texts: dict {track_id -> text} lưu QR đã decode OK.
      Các frame sau của cùng track sẽ KHÔNG decode lại — tránh NG flash
      do một frame mờ giữa chừng.
    """

    def __init__(self, model_path: str = "models/best.pt", confidence_threshold: float = 0.50,
                 infer_imgsz: int = 320, decode_padding: float = 0.15):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.infer_imgsz = infer_imgsz
        self.decode_padding = decode_padding
        self._model = None
        self._loaded = False
        self._device = self._pick_device()
        self._use_half = False
        # Cache: track_id -> text đã decode OK
        # Được set từ bên ngoài (gui.py) sau khi tracker xác nhận track_id
        # Hoặc set nội bộ nếu detect() được gọi với track_id trong det
        self.confirmed_texts: Dict[int, str] = {}

    def _pick_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detected: {name}")
                return "cuda:0"
        except Exception:
            pass
        logger.info("No GPU found, using CPU.")
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}")
            self._loaded = False
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

            if self._device.startswith("cuda"):
                try:
                    self._model.model.half()
                    self._use_half = True
                    logger.info("FP16 enabled.")
                except Exception:
                    self._use_half = False

            dummy = np.zeros((self.infer_imgsz, self.infer_imgsz, 3), dtype=np.uint8)
            self._model(dummy, verbose=False, device=self._device,
                        imgsz=self.infer_imgsz, half=self._use_half)
            logger.info(f"Model warmed up on {self._device} (imgsz={self.infer_imgsz}).")

            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._loaded = False
            return False

    def _decode_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """Crop vùng box (có thêm margin), upscale nếu cần, rồi giải mã bằng zxing-cpp.
        Trả về (text, status) với status là 'OK' hoặc 'NG'."""
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * self.decode_padding), int(bh * self.decode_padding)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
        if cx2 <= cx1 or cy2 <= cy1:
            return "", "NG"

        crop = frame[cy1:cy2, cx1:cx2]

        # Upscale nếu crop quá nhỏ — cải thiện tỉ lệ decode thành công
        # đặc biệt khi QR nằm xa camera hoặc infer_imgsz nhỏ.
        short_side = min(crop.shape[:2])
        if short_side < MIN_CROP_SHORT_SIDE and short_side > 0:
            scale = MIN_CROP_SHORT_SIDE / short_side
            new_w = int(crop.shape[1] * scale)
            new_h = int(crop.shape[0] * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        try:
            results = zxingcpp.read_barcodes(crop)
        except Exception as e:
            logger.error(f"Decode error: {e}")
            return "", "NG"

        if results:
            return results[0].text, "OK"
        return "", "NG"

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Luôn decode lại từng box phát hiện được — KHÔNG cache/skip theo vị trí.

        Lưu ý: trước đây có thử tối ưu tốc độ bằng cách bỏ qua decode nếu box
        trùng vị trí với 1 QR đã confirmed OK ở frame trước. Cách đó RỦI RO
        trên băng chuyền: nếu 1 vật mới (có thể đang lỗi) đi tới đúng vị trí
        mà vật cũ (đã OK) vừa rời đi vài frame trước, hệ thống sẽ nhầm là
        cùng 1 mã và gán OK sai mà không decode lại. Vì đây là hệ thống QA,
        độ chính xác quan trọng hơn tốc độ nên đã bỏ tối ưu này."""
        if not self._loaded or self._model is None:
            return []
        try:
            results = self._model(
                frame,
                verbose=False,
                device=self._device,
                imgsz=self.infer_imgsz,
                half=self._use_half,
                conf=self.confidence_threshold,
            )[0]
            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                text, status = self._decode_box(frame, x1, y1, x2, y2)
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf,
                    "status": status,
                    "text": text,
                })
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def update_threshold(self, threshold: float):
        self.confidence_threshold = threshold

    def update_imgsz(self, imgsz: int):
        self.infer_imgsz = imgsz

    def update_model(self, model_path: str) -> bool:
        self.model_path = model_path
        self.confirmed_texts.clear()
        return self.load()