import os
import shutil
import glob
import yaml
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DATASET_YAML_PATH = "dataset/dataset.yaml"
TRAIN_IMAGES_DIR = "train_images"
DATASET_IMAGES_DIR = "dataset/images/train"
DATASET_LABELS_DIR = "dataset/labels/train"
MODEL_OUTPUT_DIR = "models"


class Trainer:
    def __init__(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[bool, str], None]] = None
    ):
        self.on_progress = on_progress
        self.on_complete = on_complete
        self._thread: Optional[threading.Thread] = None

    def _log(self, msg: str):
        logger.info(msg)
        if self.on_progress:
            self.on_progress(msg)

    def validate_dataset(self) -> tuple[bool, str]:
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images += glob.glob(os.path.join(TRAIN_IMAGES_DIR, ext))

        if not images:
            return False, "No images found in train_images/. Please add images and annotate them."

        labeled = 0
        for img_path in images:
            label_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                labeled += 1

        if labeled == 0:
            return False, "No labeled images found. Please annotate images using the annotation tool."

        return True, f"Found {labeled}/{len(images)} labeled images."

    def build_dataset(self):
        os.makedirs(DATASET_IMAGES_DIR, exist_ok=True)
        os.makedirs(DATASET_LABELS_DIR, exist_ok=True)

        for f in glob.glob(os.path.join(DATASET_IMAGES_DIR, "*")):
            os.remove(f)
        for f in glob.glob(os.path.join(DATASET_LABELS_DIR, "*")):
            os.remove(f)

        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images += glob.glob(os.path.join(TRAIN_IMAGES_DIR, ext))

        copied = 0
        for img_path in images:
            label_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(label_path):
                continue
            shutil.copy(img_path, DATASET_IMAGES_DIR)
            shutil.copy(label_path, DATASET_LABELS_DIR)
            copied += 1

        self._log(f"Dataset built: {copied} image-label pairs.")

    def create_yaml(self):
        abs_path = os.path.abspath("dataset")
        data = {
            "path": abs_path,
            "train": "images/train",
            "val": "images/train",
            "nc": 1,
            "names": {0: "qr"}
        }
        os.makedirs("dataset", exist_ok=True)
        with open(DATASET_YAML_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        self._log(f"dataset.yaml created.")

    def train(self, epochs: int = 30, imgsz: int = 320):
        self._thread = threading.Thread(
            target=self._train_worker,
            args=(epochs, imgsz),
            daemon=True
        )
        self._thread.start()

    def _find_best_pt(self, project_dir: str) -> Optional[str]:
        """
        Search recursively under project_dir for best.pt.
        YOLO sometimes creates runs/detect/train/weights/best.pt
        or models/train/weights/best.pt depending on version.
        """
        self._log(f"Searching for best.pt under: {os.path.abspath(project_dir)}")
        for root, dirs, files in os.walk(project_dir):
            if "best.pt" in files:
                found = os.path.join(root, "best.pt")
                self._log(f"Found: {found}")
                return found

        # Also check runs/ directory (YOLO default)
        for root, dirs, files in os.walk("runs"):
            if "best.pt" in files:
                found = os.path.join(root, "best.pt")
                self._log(f"Found in runs/: {found}")
                return found

        return None

    def _train_worker(self, epochs: int, imgsz: int):
        try:
            # 1. Validate
            self._log("Validating dataset...")
            ok, msg = self.validate_dataset()
            if not ok:
                self._log(f"ERROR: {msg}")
                if self.on_complete:
                    self.on_complete(False, msg)
                return
            self._log(msg)

            # 2. Build dataset
            self._log("Building dataset structure...")
            self.build_dataset()

            # 3. Create YAML
            self._log("Creating dataset.yaml...")
            self.create_yaml()

            # 4. Detect device
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self._log(f"Using device: {'GPU (CUDA)' if device == '0' else 'CPU'}")
            except Exception:
                device = "cpu"
                self._log("Using device: CPU")

            # 5. Train
            self._log(f"Starting YOLOv8n training ({epochs} epochs, imgsz={imgsz})...")
            from ultralytics import YOLO

            model = YOLO("yolov8n.pt")

            result = model.train(
                data=os.path.abspath(DATASET_YAML_PATH),
                epochs=epochs,
                imgsz=imgsz,
                project="models",
                name="train",
                exist_ok=True,
                device=device,
                verbose=True,
                workers=0,
                batch=4,
                cache=False,
            )

            self._log("Training loop finished. Locating best.pt...")

            # 6. Find best.pt (search broadly)
            os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
            best_src = self._find_best_pt("models")

            if best_src is None:
                # Try last.pt as fallback
                last_src = None
                for root, dirs, files in os.walk("models"):
                    if "last.pt" in files:
                        last_src = os.path.join(root, "last.pt")
                        break
                if last_src:
                    self._log(f"best.pt not found, using last.pt: {last_src}")
                    best_src = last_src
                else:
                    self._log("Neither best.pt nor last.pt found after training.")
                    if self.on_complete:
                        self.on_complete(False, "best.pt not found after training.")
                    return

            best_dst = os.path.join(MODEL_OUTPUT_DIR, "best.pt")
            if os.path.abspath(best_src) != os.path.abspath(best_dst):
                shutil.copy(best_src, best_dst)

            self._log(f"Model saved to {best_dst}")
            if self.on_complete:
                self.on_complete(True, best_dst)

        except Exception as e:
            msg = f"Training failed: {e}"
            self._log(msg)
            logger.exception("Training error")
            if self.on_complete:
                self.on_complete(False, msg)