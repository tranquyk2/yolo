"""
Chạy: python test_model.py
Kiểm tra model có detect được QR không.
"""
from ultralytics import YOLO
import os
import glob

model = YOLO("models/best.pt")

# Tìm ảnh đầu tiên trong train_images/
images = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    images += glob.glob(os.path.join("train_images", ext))

if not images:
    print("Không tìm thấy ảnh trong train_images/")
    exit()

print(f"Test trên {len(images)} ảnh...\n")

total_det = 0
for img_path in images[:5]:  # test 5 ảnh đầu
    results = model(img_path, imgsz=640, verbose=False)
    n = len(results[0].boxes)
    total_det += n
    print(f"  {os.path.basename(img_path):20s} → {n} QR detected")

print(f"\nTổng: {total_det} detections trên {min(5, len(images))} ảnh")

if total_det == 0:
    print("\n❌ Model không detect được gì → cần train lại với nhiều ảnh hơn")
else:
    print("\n✅ Model hoạt động tốt → vấn đề nằm ở inference settings")