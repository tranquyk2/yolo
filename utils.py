import os
import time


def ensure_dirs():
    """Ensure all required directories exist."""
    dirs = [
        "models",
        "train_images",
        "dataset",
        "dataset/images/train",
        "dataset/labels/train"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


class FPSCounter:
    """Simple FPS counter using rolling average."""

    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))