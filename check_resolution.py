"""
Chạy: python check_resolution.py
In ra 3 con số cần để biết tại sao infer_imgsz=320 không bắt được QR
nhưng 640 lại bắt được:
  1. Size ảnh trong train_images/ (ảnh dùng để train)
  2. Frame size thật khi mở camera KIỂU debug_detect.py (không set resolution)
  3. Frame size thật khi mở camera KIỂU camera.py (ép set 1280x960)
"""
import glob
import os
import cv2
from config import load_config

cfg = load_config()
cam_idx = cfg.get("camera_index", 0)

print("=" * 60)
print("1) SIZE ẢNH TRAIN (train_images/)")
print("=" * 60)
imgs = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    imgs += glob.glob(os.path.join("train_images", ext))
if not imgs:
    print("Không tìm thấy ảnh nào trong train_images/")
else:
    # In ra vài ảnh đầu để chắc chắn tất cả cùng kích thước
    for p in sorted(imgs)[:5]:
        im = cv2.imread(p)
        if im is not None:
            h, w = im.shape[:2]
            print(f"  {os.path.basename(p):30s} -> {w}x{h}")

print()
print("=" * 60)
print("2) FRAME SIZE KIỂU debug_detect.py (KHÔNG set resolution)")
print("=" * 60)
cap = cv2.VideoCapture(cam_idx)
if not cap.isOpened():
    print("  Không mở được camera.")
else:
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"  Frame thật: {w}x{h}")
    cap.release()

print()
print("=" * 60)
print("3) FRAME SIZE KIỂU camera.py (ép set 1280x960)")
print("=" * 60)
cap2 = cv2.VideoCapture(cam_idx)
if not cap2.isOpened():
    print("  Không mở được camera.")
else:
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    ret, frame = cap2.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"  Frame thật sau khi ép 1280x960: {w}x{h}")
        if (w, h) != (1280, 960):
            print("  -> Camera KHÔNG chấp nhận đúng 1280x960, tự fallback về size khác!")
    cap2.release()

print()
print("So sánh 3 con số trên rồi gửi lại kết quả để xác định nguyên nhân chính xác.")