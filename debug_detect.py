"""
Chạy: python debug_detect.py
Test YOLO trực tiếp, không qua GUI, in ra kết quả detect từng frame.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from config import load_config

def main():
    cfg = load_config()
    model_path = cfg.get("model_path", "models/best.pt")
    cam_idx    = cfg.get("camera_index", 0)
    threshold  = cfg.get("confidence_threshold", 0.5)

    print(f"Model: {model_path}")
    print(f"Camera: {cam_idx}")
    print(f"Threshold: {threshold}")
    print()

    # Load model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model không tồn tại: {model_path}")
        return

    from ultralytics import YOLO
    print("Loading model...")
    model = YOLO(model_path)
    print("Model loaded OK\n")

    # Open camera
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được camera {cam_idx}")
        return
    print("Camera OK. Nhấn Q để thoát.\n")

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Không đọc được frame")
            continue

        frame_n += 1

        # Chạy YOLO với conf=0.1 (thấp nhất) để xem model có detect gì không
        results = model(frame, verbose=False, imgsz=640, conf=0.1)[0]

        boxes = results.boxes
        if len(boxes) == 0:
            if frame_n % 30 == 0:
                print(f"Frame {frame_n}: KHÔNG detect được gì (conf>=0.1)")
        else:
            for box in boxes:
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = model.names[cls] if hasattr(model, 'names') else str(cls)
                print(f"Frame {frame_n}: DETECT  class={label}  conf={conf:.3f}  box=({x1},{y1},{x2},{y2})")

                color = (0,255,0) if conf >= threshold else (0,165,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Debug Detect (Q=quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()