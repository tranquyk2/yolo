import json
import os

CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "camera_index": 0,
    "confidence_threshold": 0.50,
    "model_path": "models/best.pt",
    "skip_frames": 1,
    "infer_imgsz": 640,
    "result_dir": "results",
    "arduino_enabled": True,   # Bật/tắt gửi lệnh xuống Arduino
    "arduino_port": "",        # Rỗng = tự động dò theo VID/PID
    "confirm_frames": 6,       # Số hit liên tiếp cần để ô xám "..." chốt thành OK/NG
                                # (số càng lớn = QR phải đi qua vùng scan lâu hơn mới bắt)
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)