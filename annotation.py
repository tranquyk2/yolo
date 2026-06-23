import os
import json
from typing import List, Tuple, Optional
import cv2
import numpy as np


# YOLO label format: class_id cx cy w h (all normalized 0-1)

def bbox_to_yolo(img_w: int, img_h: int, x1: int, y1: int, x2: int, y2: int) -> str:
    """Convert pixel bbox to YOLO format string (class 0 = qr)."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def save_yolo_labels(label_path: str, img_w: int, img_h: int, boxes: List[Tuple[int, int, int, int]]):
    """Save a list of (x1, y1, x2, y2) boxes to a YOLO .txt label file."""
    lines = [bbox_to_yolo(img_w, img_h, *box) for box in boxes]
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    """Load YOLO labels and convert back to pixel coords."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes


class AnnotationSession:
    """
    Manages annotation state for a single image.
    Tracks current boxes and supports undo.
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        self.h, self.w = self.image.shape[:2]

        self.label_path = os.path.splitext(image_path)[0] + ".txt"
        self.boxes: List[Tuple[int, int, int, int]] = load_yolo_labels(self.label_path, self.w, self.h)

        self.drawing = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.current_pt: Optional[Tuple[int, int]] = None

    def begin_draw(self, x: int, y: int):
        self.drawing = True
        self.start_pt = (x, y)
        self.current_pt = (x, y)

    def update_draw(self, x: int, y: int):
        if self.drawing:
            self.current_pt = (x, y)

    def end_draw(self, x: int, y: int) -> bool:
        """Finalize a bounding box. Returns True if box is valid (non-trivial size)."""
        if not self.drawing or self.start_pt is None:
            return False
        self.drawing = False
        x1 = min(self.start_pt[0], x)
        y1 = min(self.start_pt[1], y)
        x2 = max(self.start_pt[0], x)
        y2 = max(self.start_pt[1], y)
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.start_pt = None
            self.current_pt = None
            return False
        self.boxes.append((x1, y1, x2, y2))
        self.start_pt = None
        self.current_pt = None
        return True

    def undo(self):
        if self.boxes:
            self.boxes.pop()

    def save(self):
        save_yolo_labels(self.label_path, self.w, self.h, self.boxes)

    def render(self) -> np.ndarray:
        """Render image with all boxes and current drawing box."""
        display = self.image.copy()
        for (x1, y1, x2, y2) in self.boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(display, "qr", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        if self.drawing and self.start_pt and self.current_pt:
            cv2.rectangle(display, self.start_pt, self.current_pt, (0, 255, 255), 1)

        return display