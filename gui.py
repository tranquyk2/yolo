import os
import sys
import subprocess
import logging
import cv2
import threading
import time
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np

from config import load_config, save_config
from camera import CameraManager
from detector import QRDetector
from qr_tracker import QRTracker
from result_logger import ResultLogger
from trainer import Trainer
from annotation import AnnotationSession
from utils import FPSCounter, ensure_dirs
from arduino_sent import ArduinoConnection, find_arduino_port

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

OK_COLOR,      OK_LABEL_BG      = (0, 220, 80),    (0, 180, 60)     # xanh lá - decode OK
NG_COLOR,      NG_LABEL_BG      = (0, 0, 230),     (0, 0, 180)      # đỏ     - decode NG
PENDING_COLOR, PENDING_LABEL_BG = (160, 160, 160), (110, 110, 110)  # xám    - đang chờ đủ hit


# ─── Annotation Window ────────────────────────────────────────────────────────
class AnnotationWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Annotation Tool – Create Dataset")
        self.geometry("1000x780")
        self.resizable(True, True)
        self._session = None
        self._image_paths = []
        self._current_idx = 0
        self._tk_img = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._build_ui()
        self._load_image_list()
        self.transient(master)
        self.lift()
        self.focus_force()

    def _build_ui(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(top, text="Annotation Tool", font=("Arial", 16, "bold")).pack(side="left")
        ctk.CTkButton(top, text="◀ Prev", width=80, command=self._prev_image).pack(side="left", padx=4)
        ctk.CTkButton(top, text="Next ▶", width=80, command=self._next_image).pack(side="left", padx=4)
        self._img_label_var = ctk.StringVar(value="No image loaded")
        ctk.CTkLabel(top, textvariable=self._img_label_var).pack(side="left", padx=12)
        ctk.CTkButton(top, text="↩ Undo", width=80, command=self._undo).pack(side="right", padx=4)
        ctk.CTkButton(top, text="💾 Save", width=100,
                      fg_color="#1a7a40", hover_color="#155c30",
                      command=self._save).pack(side="right", padx=4)
        canvas_frame = ctk.CTkFrame(self)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", cursor="crosshair")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self._status_var = ctk.StringVar(value="Open images from train_images/ and draw boxes around QR codes.")
        ctk.CTkLabel(self, textvariable=self._status_var, text_color="#aaaaaa").pack(pady=(0, 6))
        ctk.CTkLabel(self, text="Tip: Drag = draw box  |  Undo = remove last  |  Save = write label",
                     text_color="#666666", font=("Arial", 11)).pack(pady=(0, 4))

    def _load_image_list(self):
        ensure_dirs()
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths += glob.glob(os.path.join("train_images", ext))
        self._image_paths = sorted(paths)
        if self._image_paths:
            self._current_idx = 0
            self._open_image(self._image_paths[0])
        else:
            self._status_var.set("No images in train_images/. Copy images there first.")

    def _open_image(self, path):
        try:
            self._session = AnnotationSession(path)
            name = os.path.basename(path)
            n = len(self._session.boxes)
            total = len(self._image_paths)
            self._img_label_var.set(f"[{self._current_idx+1}/{total}]  {name}  ({n} labels)")
            self._status_var.set(f"Loaded: {name}")
            self._render()
        except Exception as e:
            self._status_var.set(f"Error: {e}")

    def _render(self):
        if self._session is None:
            return
        frame = self._session.render()
        h, w = frame.shape[:2]
        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)
        scale = min(cw / w, ch / h, 1.0)
        self._scale = scale
        nw, nh = int(w * scale), int(h * scale)
        self._offset_x = (cw - nw) // 2
        self._offset_y = (ch - nh) // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(self._offset_x, self._offset_y, anchor="nw", image=self._tk_img)
        n = len(self._session.boxes)
        self._canvas.create_text(10, 10, anchor="nw",
                                  text=f"  {n} QR box{'es' if n!=1 else ''}  ",
                                  fill="white", font=("Arial", 12, "bold"))

    def _canvas_to_image(self, cx, cy):
        if self._session is None:
            return 0, 0
        ix = int((cx - self._offset_x) / self._scale)
        iy = int((cy - self._offset_y) / self._scale)
        ix = max(0, min(self._session.w - 1, ix))
        iy = max(0, min(self._session.h - 1, iy))
        return ix, iy

    def _on_mouse_down(self, event):
        if self._session:
            self._session.begin_draw(*self._canvas_to_image(event.x, event.y))

    def _on_mouse_drag(self, event):
        if self._session:
            self._session.update_draw(*self._canvas_to_image(event.x, event.y))
            self._render()

    def _on_mouse_up(self, event):
        if self._session:
            added = self._session.end_draw(*self._canvas_to_image(event.x, event.y))
            if added:
                n = len(self._session.boxes)
                self._status_var.set(f"{n} box{'es' if n!=1 else ''} drawn — remember to Save.")
            self._render()

    def _undo(self):
        if self._session:
            self._session.undo()
            self._status_var.set("Last box removed.")
            self._render()

    def _save(self):
        if self._session:
            self._session.save()
            n = len(self._session.boxes)
            name = os.path.basename(self._session.image_path)
            self._status_var.set(f"Saved {n} label(s) for {name}")

    def _prev_image(self):
        if self._image_paths:
            self._current_idx = (self._current_idx - 1) % len(self._image_paths)
            self._open_image(self._image_paths[self._current_idx])

    def _next_image(self):
        if self._image_paths:
            self._current_idx = (self._current_idx + 1) % len(self._image_paths)
            self._open_image(self._image_paths[self._current_idx])


# ─── Training Window ──────────────────────────────────────────────────────────
class TrainingWindow(ctk.CTkToplevel):
    def __init__(self, master, on_trained_callback=None):
        super().__init__(master)
        self.title("Train Model")
        self.geometry("680x520")
        self.resizable(False, False)
        self._on_trained = on_trained_callback
        self._training = False
        self._build_ui()
        self.transient(master)
        self.lift()
        self.focus_force()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Train YOLOv8 Model", font=("Arial", 17, "bold")).pack(pady=16)
        settings = ctk.CTkFrame(self)
        settings.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(settings, text="Epochs:").grid(row=0, column=0, padx=12, pady=8, sticky="w")
        self._epochs_var = ctk.IntVar(value=50)
        ctk.CTkSlider(settings, from_=10, to=200, number_of_steps=19,
                      variable=self._epochs_var, width=220).grid(row=0, column=1, padx=8)
        self._epochs_disp = ctk.CTkLabel(settings, text="50")
        self._epochs_disp.grid(row=0, column=2, padx=8)
        self._epochs_var.trace_add("write", lambda *_: self._epochs_disp.configure(
            text=str(self._epochs_var.get())))
        ctk.CTkLabel(settings, text="Image Size:").grid(row=1, column=0, padx=12, pady=8, sticky="w")
        self._imgsz_var = ctk.StringVar(value="640")
        ctk.CTkOptionMenu(settings, variable=self._imgsz_var,
                          values=["320", "416", "512", "640"]).grid(row=1, column=1, padx=8)
        self._log_box = ctk.CTkTextbox(self, height=260, font=("Courier", 11))
        self._log_box.pack(fill="both", expand=True, padx=20, pady=4)
        self._progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self._progress.pack(fill="x", padx=20, pady=4)
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=10)
        self._train_btn = ctk.CTkButton(
            btn_row, text="▶  TRAIN MODEL", width=160,
            fg_color="#1a7a40", hover_color="#155c30",
            font=("Arial", 13, "bold"),
            command=self._start_training)
        self._train_btn.pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="Close", width=100, command=self.destroy).pack(side="left", padx=8)

    def _log(self, msg):
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")

    def _start_training(self):
        if self._training:
            return
        self._training = True
        self._train_btn.configure(state="disabled", text="Training…")
        self._progress.start()
        self._log_box.delete("1.0", "end")
        self._log("Starting training pipeline...\n")
        trainer = Trainer(on_progress=self._on_progress, on_complete=self._on_complete)
        trainer.train(epochs=self._epochs_var.get(), imgsz=int(self._imgsz_var.get()))

    def _on_progress(self, msg):
        self.after(0, self._log, msg)

    def _on_complete(self, success, info):
        def _finish():
            self._progress.stop()
            self._training = False
            self._train_btn.configure(state="normal", text="▶  TRAIN MODEL")
            if success:
                self._log(f"\n✅ Training complete! Model saved to: {info}")
                messagebox.showinfo("Training Complete", f"Model saved to:\n{info}", parent=self)
                if self._on_trained:
                    self._on_trained(info)
            else:
                self._log(f"\n❌ Training failed: {info}")
                messagebox.showerror("Training Failed", info, parent=self)
        self.after(0, _finish)


# ─── Settings Dialog ──────────────────────────────────────────────────────────
class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master, config, on_save):
        super().__init__(master)
        self.title("Settings")
        self.geometry("520x680")
        self.resizable(False, False)
        self._config = config.copy()
        self._on_save = on_save
        self._build_ui()
        self.transient(master)        # gắn với cửa sổ chính, luôn nổi trên nó
        self.lift()
        self.focus_force()
        self.grab_set()               # modal: chặn thao tác cửa sổ chính khi đang mở Settings
        self.attributes("-topmost", True)
        self.after(150, lambda: self.attributes("-topmost", False))

    def _build_ui(self):
        ctk.CTkLabel(self, text="⚙  Settings", font=("Arial", 16, "bold")).pack(pady=14)
        form = ctk.CTkFrame(self)
        form.pack(fill="x", padx=24, pady=8)
        form.columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Camera Index:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self._cam_var = ctk.StringVar(value=str(self._config.get("camera_index", 0)))
        ctk.CTkEntry(form, textvariable=self._cam_var, width=80).grid(row=0, column=1, padx=10, sticky="w")

        ctk.CTkLabel(form, text="Confidence:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self._conf_var = ctk.DoubleVar(value=self._config.get("confidence_threshold", 0.50))
        conf_row = ctk.CTkFrame(form, fg_color="transparent")
        conf_row.grid(row=1, column=1, padx=10, sticky="ew")
        ctk.CTkSlider(conf_row, from_=0.1, to=0.95, variable=self._conf_var, width=180).pack(side="left")
        self._conf_lbl = ctk.CTkLabel(conf_row, text=f"{self._conf_var.get():.2f}", width=46)
        self._conf_lbl.pack(side="left", padx=6)
        self._conf_var.trace_add("write", lambda *_: self._conf_lbl.configure(
            text=f"{self._conf_var.get():.2f}"))

        ctk.CTkLabel(form, text="Skip Frames:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self._skip_var = ctk.StringVar(value=str(self._config.get("skip_frames", 1)))
        skip_row = ctk.CTkFrame(form, fg_color="transparent")
        skip_row.grid(row=2, column=1, padx=10, sticky="w")
        ctk.CTkOptionMenu(skip_row, variable=self._skip_var,
                          values=["1", "2", "3", "4"], width=80).pack(side="left")
        ctk.CTkLabel(skip_row, text="  (1=mọi frame)", text_color="#888").pack(side="left", padx=6)

        ctk.CTkLabel(form, text="Model Path:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self._model_var = ctk.StringVar(value=self._config.get("model_path", "models/best.pt"))
        model_row = ctk.CTkFrame(form, fg_color="transparent")
        model_row.grid(row=3, column=1, padx=10, sticky="ew")
        ctk.CTkEntry(model_row, textvariable=self._model_var, width=200).pack(side="left")
        ctk.CTkButton(model_row, text="Browse…", width=72,
                      command=self._browse_model).pack(side="left", padx=6)

        ctk.CTkLabel(form, text="Inference Size:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self._imgsz_var = ctk.StringVar(value=str(self._config.get("infer_imgsz", 320)))
        imgsz_row = ctk.CTkFrame(form, fg_color="transparent")
        imgsz_row.grid(row=4, column=1, padx=10, sticky="w")
        ctk.CTkOptionMenu(imgsz_row, variable=self._imgsz_var,
                          values=["160", "224", "320", "416", "640"], width=80).pack(side="left")
        ctk.CTkLabel(imgsz_row, text="  (nhỏ hơn = nhanh hơn)", text_color="#888").pack(side="left", padx=6)

        ctk.CTkLabel(form, text="Thư mục kết quả:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self._result_dir_var = ctk.StringVar(value=self._config.get("result_dir", "results"))
        result_row = ctk.CTkFrame(form, fg_color="transparent")
        result_row.grid(row=5, column=1, padx=10, sticky="ew")
        ctk.CTkEntry(result_row, textvariable=self._result_dir_var, width=160).pack(side="left")
        ctk.CTkButton(result_row, text="Browse…", width=72,
                      command=self._browse_result_dir).pack(side="left", padx=6)

        # ── Quãng đường xác nhận (px) — TỰ ĐỘNG thích nghi tốc độ băng chuyền ──
        ctk.CTkLabel(form, text="Quãng đường xác nhận:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self._confirm_distance_var = ctk.IntVar(value=self._config.get("confirm_distance_px", 150))
        confirm_row = ctk.CTkFrame(form, fg_color="transparent")
        confirm_row.grid(row=6, column=1, padx=10, sticky="ew")
        ctk.CTkSlider(confirm_row, from_=20, to=500, number_of_steps=48,
                      variable=self._confirm_distance_var, width=160).pack(side="left")
        self._confirm_distance_disp = ctk.CTkLabel(confirm_row, text=f"{self._confirm_distance_var.get()}px", width=40)
        self._confirm_distance_disp.pack(side="left", padx=6)
        self._confirm_distance_var.trace_add("write", lambda *_: self._confirm_distance_disp.configure(
            text=f"{self._confirm_distance_var.get()}px"))
        ctk.CTkLabel(form, text="(đo bằng khoảng cách QR di chuyển trong khung hình,\nKHÔNG phải số frame — tự động thích nghi khi tốc độ\nbăng chuyền đổi, ví dụ cuộn tem to dần lên, không\ncần chỉnh tay lại)",
                     text_color="#666666", font=("Arial", 10), justify="left").grid(
            row=7, column=1, padx=10, sticky="w")

        # ── Arduino ──────────────────────────────────────────────────────────
        ctk.CTkLabel(form, text="Arduino:", font=("Arial", 12, "bold")).grid(
            row=8, column=0, padx=10, pady=(20, 4), sticky="w")
        self._arduino_enabled_var = ctk.BooleanVar(value=self._config.get("arduino_enabled", True))
        ctk.CTkCheckBox(form, text="Bật điều khiển motor (gửi lệnh khi NG)",
                        variable=self._arduino_enabled_var).grid(
            row=8, column=1, padx=10, pady=(20, 4), sticky="w")

        ctk.CTkLabel(form, text="Cổng COM:").grid(row=9, column=0, padx=10, pady=10, sticky="w")
        arduino_row = ctk.CTkFrame(form, fg_color="transparent")
        arduino_row.grid(row=9, column=1, padx=10, sticky="ew")

        AUTO_LABEL = "Tự động dò"
        saved_port = self._config.get("arduino_port", "")
        self._arduino_port_var = ctk.StringVar(value=saved_port if saved_port else AUTO_LABEL)
        self._arduino_port_values = [AUTO_LABEL]  # sẽ được _refresh_ports() lấp đầy

        self._arduino_port_menu = ctk.CTkOptionMenu(
            arduino_row, variable=self._arduino_port_var,
            values=self._arduino_port_values, width=160)
        self._arduino_port_menu.pack(side="left")
        ctk.CTkButton(arduino_row, text="⟳", width=32,
                      command=self._refresh_ports).pack(side="left", padx=4)
        ctk.CTkButton(arduino_row, text="Test", width=56,
                      command=self._test_arduino).pack(side="left", padx=4)
        self._arduino_test_lbl = ctk.CTkLabel(arduino_row, text="", text_color="#888")
        self._arduino_test_lbl.pack(side="left", padx=4)

        ctk.CTkLabel(form, text="(\"Tự động dò\" = nhận diện theo VID/PID; bấm ⟳ để quét lại cổng)",
                     text_color="#666666", font=("Arial", 10)).grid(
            row=10, column=1, padx=10, sticky="w")

        self._refresh_ports(keep_selection=True)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=20)
        ctk.CTkButton(btn_row, text="💾  Save", width=130,
                      fg_color="#1a7a40", hover_color="#155c30",
                      font=("Arial", 13, "bold"),
                      command=self._save).pack(side="left", padx=10)
        ctk.CTkButton(btn_row, text="Cancel", width=100, command=self.destroy).pack(side="left", padx=10)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            parent=self, title="Select model file",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if path:
            self._model_var.set(path)

    def _browse_result_dir(self):
        path = filedialog.askdirectory(parent=self, title="Chọn thư mục lưu kết quả OK/NG")
        if path:
            self._result_dir_var.set(path)

    def _refresh_ports(self, keep_selection: bool = False):
        """Quét lại danh sách cổng COM thật trên máy và nạp vào dropdown."""
        from arduino_sent import list_serial_ports
        AUTO_LABEL = "Tự động dò"
        ports = list_serial_ports()
        values = [AUTO_LABEL] + ports
        self._arduino_port_menu.configure(values=values)

        current = self._arduino_port_var.get()
        if keep_selection and current in values:
            return  # giữ nguyên lựa chọn đã lưu trong config nếu vẫn còn tồn tại
        if current not in values:
            self._arduino_port_var.set(AUTO_LABEL)

    def _test_arduino(self):
        AUTO_LABEL = "Tự động dò"
        selected = self._arduino_port_var.get()
        port = None if selected == AUTO_LABEL else selected
        port = port or find_arduino_port()
        if port:
            self._arduino_test_lbl.configure(text=f"Tìm thấy: {port}", text_color="#1a7a40")
        else:
            self._arduino_test_lbl.configure(text="Không tìm thấy thiết bị", text_color="#c0392b")

    def _save(self):
        try:
            self._config["camera_index"] = int(self._cam_var.get())
        except ValueError:
            messagebox.showerror("Error", "Camera index phải là số nguyên.", parent=self)
            return
        self._config["confidence_threshold"] = round(self._conf_var.get(), 2)
        self._config["skip_frames"] = int(self._skip_var.get())
        self._config["infer_imgsz"] = int(self._imgsz_var.get())
        self._config["model_path"] = self._model_var.get()
        self._config["result_dir"] = self._result_dir_var.get()
        self._config["confirm_distance_px"] = int(self._confirm_distance_var.get())
        self._config["arduino_enabled"] = self._arduino_enabled_var.get()
        selected_port = self._arduino_port_var.get()
        self._config["arduino_port"] = "" if selected_port == "Tự động dò" else selected_port
        self._on_save(self._config)
        self.destroy()


# ─── Main Application ─────────────────────────────────────────────────────────
class MainApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("QR Detection System  –  YOLOv8 + GPU")
        self.geometry("1280x800")
        self.minsize(1050, 700)

        ensure_dirs()
        self._config = load_config()
        self._fps = FPSCounter(window=30)

        self._camera = CameraManager(
            camera_index=self._config["camera_index"],
            on_status_change=self._on_camera_status)

        self._detector = QRDetector(
            model_path=self._config["model_path"],
            confidence_threshold=self._config["confidence_threshold"],
            infer_imgsz=self._config.get("infer_imgsz", 320))

        # ResultLogger: lưu CSV (OK) và ảnh NG vào thư mục kết quả
        self._result_logger = ResultLogger(
            output_dir=self._config.get("result_dir", "results"))

        # Tracker v6: IoU matching + distance-fallback + tự động thích nghi
        # tốc độ băng chuyền (chốt NG theo quãng đường px, không theo số frame)
        self._tracker = QRTracker(
            max_misses=15,
            confirm_distance_px=self._config.get("confirm_distance_px", 150),
            on_finalize=self._on_item_finalized,
        )
        # Cache track đã confirmed OK → tránh NG flash do 1 frame mờ
        self._confirmed_ok: dict = {}
        # Set track_id đã gửi lệnh NG xuống Arduino (tránh gửi lặp lại nhiều lần
        # cho cùng 1 QR trong khi nó vẫn còn trong khung hình)
        self._ng_triggered: set = set()
        # ── Khoá "còn tem NG" ────────────────────────────────────────────────
        # motor_locked=True nghĩa là đã có ít nhất 1 lần NG chưa được CLEAR.
        # Chỉ khi vùng scan không còn detection NG nào trong nhiều frame liên
        # tiếp mới gửi CLEAR xuống Arduino để mở khoá nút nhấn.
        self._motor_locked = False
        self._empty_ng_streak = 0
        self._NG_CLEAR_DEBOUNCE_FRAMES = 10  # điều chỉnh theo tốc độ băng chuyền

        # ── Arduino (motor + còi) ────────────────────────────────────────────
        # arduino_status_var: hiển thị trạng thái kết nối lên Live Stats
        self._arduino_status_var = ctk.StringVar(value="—")
        self._arduino: ArduinoConnection | None = None
        self._init_arduino()

        self._total_scanned = 0
        self._total_ok = 0
        self._total_ng = 0

        # Frame được render sẵn (đã vẽ box) từ detection thread
        self._display_frame = None
        self._display_lock = threading.Lock()

        self._det_running = False
        self._detection_thread = None
        self._frame_counter = 0
        self._skip_frames = self._config.get("skip_frames", 1)
        self._camera_status = "Not started"
        self._model_status = "Not loaded"
        self._qr_count = 0
        self._ok_count = 0
        self._ng_count = 0
        self._fps_val = 0.0

        # Lưu tk image tránh GC
        self._tk_img_ref = None
        # ID của canvas image item — tái dùng thay vì delete("all") mỗi frame
        self._canvas_img_id = None
        # Cờ chống dồn frame
        self._pending_draw = False

        # Trạng thái chụp ảnh để build dataset
        self._capture_status_var = ctk.StringVar(value="Chưa chụp ảnh nào.")

        self._build_ui()
        self._try_load_model()
        self._poll_stats()
        self.bind("<KeyPress-c>", lambda e: self._capture_training_image())
        self.bind("<KeyPress-C>", lambda e: self._capture_training_image())

    # ── Arduino ───────────────────────────────────────────────────────────────
    def _init_arduino(self):
        """Khởi tạo / khởi tạo lại kết nối Arduino theo config hiện tại.
        Chạy trong thread riêng để không block UI khi mở cổng serial
        (Arduino reset mất ~2s khi mở cổng)."""
        # Đóng kết nối cũ nếu có (vd khi user đổi cổng/tắt trong Settings)
        if self._arduino is not None:
            try:
                self._arduino.close()
            except Exception:
                pass
            self._arduino = None

        if not self._config.get("arduino_enabled", True):
            self._arduino_status_var.set("Đã tắt")
            return

        self._arduino_status_var.set("Đang kết nối...")
        port_cfg = self._config.get("arduino_port", "")

        def _connect():
            try:
                ard = ArduinoConnection(
                    port=port_cfg if port_cfg else None,
                    auto_detect=True,
                    auto_reconnect=True,
                )
            except Exception as e:
                logger.error(f"Lỗi khởi tạo Arduino: {e}")
                self.after(0, lambda: self._arduino_status_var.set("Lỗi kết nối"))
                return
            self._arduino = ard
            status = f"Connected ({ard.port})" if ard.is_connected() else "Không tìm thấy"
            self.after(0, lambda: self._arduino_status_var.set(status))

        threading.Thread(target=_connect, daemon=True).start()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        left = ctk.CTkFrame(self)
        left.pack(side="left", fill="both", expand=True, padx=(10, 4), pady=10)

        title_row = ctk.CTkFrame(left, fg_color="transparent")
        title_row.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(title_row, text="🔍  QR Detection System",
                     font=("Arial", 18, "bold")).pack(side="left")

        self._canvas = tk.Canvas(left, bg="#0d0d0d", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=8, pady=6)
        self._canvas.create_text(400, 300,
                                  text="Camera feed will appear here\nPress  Start Camera  to begin",
                                  fill="#444444", font=("Arial", 15), justify="center")

        right = ctk.CTkScrollableFrame(self, width=230, fg_color="transparent")
        right.pack(side="right", fill="y", padx=(4, 10), pady=10)

        ctk.CTkLabel(right, text="Controls", font=("Arial", 14, "bold")).pack(pady=(14, 6))

        # ── Camera ────────────────────────────────────────────────────────────
        cam_frame = ctk.CTkFrame(right)
        cam_frame.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(cam_frame, text="Camera", font=("Arial", 12, "bold")).pack(pady=(6, 4))
        self._start_btn = ctk.CTkButton(
            cam_frame, text="▶  Start Camera", width=190,
            fg_color="#1a5276", hover_color="#154360",
            command=self._start_camera)
        self._start_btn.pack(pady=4)
        self._stop_btn = ctk.CTkButton(
            cam_frame, text="⏹  Stop Camera", width=190,
            fg_color="#641e16", hover_color="#4a1510",
            state="disabled", command=self._stop_camera)
        self._stop_btn.pack(pady=4)

        # ── Training ──────────────────────────────────────────────────────────
        train_frame = ctk.CTkFrame(right)
        train_frame.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(train_frame, text="Training", font=("Arial", 12, "bold")).pack(pady=(6, 4))
        self._capture_btn = ctk.CTkButton(
            train_frame, text="📷  Chụp ảnh Train", width=190,
            fg_color="#7d5a00", hover_color="#5c4200",
            state="disabled", command=self._capture_training_image)
        self._capture_btn.pack(pady=4)
        ctk.CTkLabel(train_frame, textvariable=self._capture_status_var,
                     text_color="#888888", font=("Arial", 10),
                     wraplength=180, justify="left").pack(pady=(0, 4), padx=4)
        ctk.CTkLabel(train_frame,
                     text="Tip: chụp ở đúng khoảng cách lúc scan thật\n(đừng zoom/crop), nhấn C để chụp nhanh.",
                     text_color="#666666", font=("Arial", 10),
                     justify="left").pack(pady=(0, 6), padx=4)
        ctk.CTkButton(train_frame, text="🏷  Create Dataset", width=190,
                      command=self._open_annotation).pack(pady=4)
        ctk.CTkButton(train_frame, text="🧠  Train Model", width=190,
                      fg_color="#1a7a40", hover_color="#155c30",
                      font=("Arial", 12, "bold"),
                      command=self._open_training).pack(pady=(4, 6))

        # ── Kết quả ───────────────────────────────────────────────────────────
        result_frame = ctk.CTkFrame(right)
        result_frame.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(result_frame, text="Kết quả", font=("Arial", 12, "bold")).pack(pady=(6, 4))
        ctk.CTkButton(
            result_frame, text="📂  Mở thư mục OK (CSV)", width=190,
            fg_color="#1a3a5c", hover_color="#112840",
            command=self._open_ok_folder).pack(pady=3)
        ctk.CTkButton(
            result_frame, text="📂  Mở thư mục ảnh NG", width=190,
            fg_color="#4a1515", hover_color="#360f0f",
            command=self._open_ng_folder).pack(pady=(3, 6))

        # ── Settings ──────────────────────────────────────────────────────────
        ctk.CTkButton(right, text="⚙  Settings", width=190,
                      command=self._open_settings).pack(pady=(4, 2))

        # ── Live Stats ────────────────────────────────────────────────────────
        stats = ctk.CTkFrame(right)
        stats.pack(fill="x", padx=10, pady=6)
        ctk.CTkLabel(stats, text="Live Stats", font=("Arial", 12, "bold")).pack(pady=(6, 6))

        def stat_row(label):
            row = ctk.CTkFrame(stats, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=2)
            ctk.CTkLabel(row, text=label, text_color="#888888", width=100, anchor="w").pack(side="left")
            var = ctk.StringVar(value="—")
            ctk.CTkLabel(row, textvariable=var, anchor="e").pack(side="right")
            return var

        self._qr_count_var     = stat_row("QR Count:")
        self._ok_count_var     = stat_row("OK:")
        self._ng_count_var     = stat_row("NG:")
        self._total_var        = stat_row("Tổng đã quét:")
        self._total_ok_var     = stat_row("Tổng OK:")
        self._total_ng_var     = stat_row("Tổng NG:")
        self._fps_var          = stat_row("FPS:")
        self._cam_status_var   = stat_row("Camera:")
        self._model_status_var = stat_row("Model:")
        self._device_var       = stat_row("Device:")
        self._conf_disp_var    = stat_row("Threshold:")
        self._arduino_stat_var = stat_row("Arduino:")
        self._lock_stat_var    = stat_row("Khoá NG:")

        self._device_var.set(self._detector._device.upper())
        self._conf_disp_var.set(f"{self._config['confidence_threshold']:.0%}")

    # ── Model ─────────────────────────────────────────────────────────────────
    def _try_load_model(self):
        def _load():
            ok = self._detector.load()
            def _ui():
                self._model_status = "Loaded ✓" if ok else "No model"
                if ok:
                    self._device_var.set("GPU" if self._detector._device.startswith("cuda") else "CPU")
            self.after(0, _ui)
        threading.Thread(target=_load, daemon=True).start()

    # ── Camera ────────────────────────────────────────────────────────────────
    def _start_camera(self):
        self._camera.camera_index = self._config["camera_index"]
        self._camera.start()
        self._det_running = True
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._capture_btn.configure(state="normal")

    def _stop_camera(self):
        self._det_running = False
        self._camera.stop()
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._capture_btn.configure(state="disabled")

    def _on_camera_status(self, status):
        self._camera_status = status

    # ── Detection loop ────────────────────────────────────────────────────────
    def _detection_loop(self):
        skip = max(1, self._skip_frames)
        last_detections = []

        while self._det_running:
            frame = self._camera.read()
            if frame is None:
                time.sleep(0.005)
                continue

            self._frame_counter += 1

            # YOLO detect + track
            if self._frame_counter % skip == 0 and self._detector.is_loaded:
                raw_dets = self._detector.detect(frame)
                last_detections = self._tracker.update(raw_dets)

                # confirmed_ok cache: nếu frame này NG nhưng track đã từng OK → ép OK
                # (chỉ áp dụng theo track_id do tracker gán, KHÔNG theo vị trí thô
                # để tránh nhầm giữa các vật khác nhau đi qua cùng vị trí màn hình)
                for det in last_detections:
                    tid = det.get("track_id")
                    if tid is None:
                        continue
                    if det.get("status") == "OK":
                        self._confirmed_ok[tid] = det.get("text", "")
                    elif tid in self._confirmed_ok:
                        det["status"] = "OK"
                        det["text"] = self._confirmed_ok[tid]

                # Cập nhật frame cuối cho ResultLogger (frame GỐC chưa vẽ box)
                for det in last_detections:
                    tid = det.get("track_id")
                    if tid is not None:
                        self._result_logger.update_last_frame(tid, frame, det)

                # ── Gửi lệnh NG xuống Arduino NGAY KHI phát hiện NG ─────────
                # Không chờ QR rời khỏi khung hình mới gửi (như _on_item_finalized
                # làm) vì lúc đó băng chuyền đã chạy qua rồi — cần dừng motor
                # ngay lập tức để tránh bỏ sót hàng lỗi.
                # Dùng _ng_triggered để đảm bảo mỗi track_id chỉ gửi đúng 1 lần,
                # không gửi lặp lại liên tục trong khi QR vẫn còn trong khung.
                for det in last_detections:
                    tid = det.get("track_id")
                    if tid is None:
                        continue
                    if det.get("status") == "NG" and tid not in self._ng_triggered:
                        self._ng_triggered.add(tid)
                        self._motor_locked = True
                        self._empty_ng_streak = 0
                        if self._arduino is not None and self._arduino.is_connected():
                            try:
                                self._arduino.send_ng()
                                logger.info(f"[Arduino] Gửi NG ngay cho track {tid}")
                            except Exception as e:
                                logger.error(f"Lỗi gửi NG Arduino: {e}")

                # ── Kiểm tra vùng scan còn tem NG nào không, để mở khoá ─────
                # Chỉ gửi CLEAR khi KHÔNG có detection NG nào trong nhiều
                # frame liên tiếp (debounce chống rung/miss-detect 1 frame).
                # Đây là cơ chế bắt buộc "bóc hết tem NG mới cho chạy lại":
                # nút vật lý trên Arduino sẽ TỪ CHỐI resume cho tới khi
                # nhận được lệnh CLEAR này từ PC.
                still_has_ng = any(d.get("status") == "NG" for d in last_detections)
                if still_has_ng:
                    self._empty_ng_streak = 0
                else:
                    self._empty_ng_streak += 1
                    if (self._motor_locked
                            and self._empty_ng_streak >= self._NG_CLEAR_DEBOUNCE_FRAMES):
                        self._motor_locked = False
                        self._ng_triggered.clear()
                        if self._arduino is not None and self._arduino.is_connected():
                            try:
                                self._arduino.send_clear()
                                logger.info("[Arduino] Đã gửi CLEAR — cho phép nhấn nút chạy lại")
                            except Exception as e:
                                logger.error(f"Lỗi gửi CLEAR Arduino: {e}")

                self._qr_count = len(last_detections)
                self._ok_count  = sum(1 for d in last_detections if d.get("status") == "OK")
                self._ng_count  = sum(1 for d in last_detections if d.get("status") == "NG")

            # Vẽ box — 3 màu: OK (xanh) | NG (đỏ) | ... (xám, đang chờ đủ hit)
            for det in last_detections:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                status = det.get("status", "...")
                if status == "OK":
                    color, label_bg = OK_COLOR, OK_LABEL_BG
                elif status == "NG":
                    color, label_bg = NG_COLOR, NG_LABEL_BG
                else:
                    color, label_bg = PENDING_COLOR, PENDING_LABEL_BG
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = status
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - lh - 12), (x1 + lw + 10, y1), label_bg, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Scale + gửi lên canvas
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw > 2 and ch > 2 and not self._pending_draw:
                h, w = frame.shape[:2]
                scale = min(cw / w, ch / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                ox = (cw - nw) // 2
                oy = (ch - nh) // 2
                self._pending_draw = True
                self.after(0, self._update_canvas_image, rgb, ox, oy)

            self._fps.tick()
            self._fps_val = self._fps.fps

    # ── Finalize callback ─────────────────────────────────────────────────────
    def _on_item_finalized(self, track):
        """Gọi (từ thread detect) đúng 1 lần khi 1 QR vật lý rời khung hình.
        Lưu kết quả (CSV hoặc ảnh NG) rồi cộng dồn bộ đếm.

        Lệnh NG đã được gửi ngay trong detection loop (khi display_status
        chuyển sang 'NG' lần đầu) nên KHÔNG gửi lại ở đây để tránh
        kích hoạt dừng motor 2 lần cho cùng 1 sản phẩm."""
        # Lưu file — chạy trong detection thread, không block UI
        self._result_logger.log(track)

        # Dọn confirmed_ok cache cho track này
        self._confirmed_ok.pop(track.id, None)
        self._ng_triggered.discard(track.id)

        def _update():
            self._total_scanned += 1
            if track.status == "OK":
                self._total_ok += 1
            else:
                self._total_ng += 1
        self.after(0, _update)

    def _update_canvas_image(self, rgb: np.ndarray, ox: int, oy: int):
        """Chạy trên main thread: tạo PhotoImage và update canvas item."""
        try:
            pil_img = Image.fromarray(rgb)
            self._tk_img_ref = ImageTk.PhotoImage(pil_img)
            if self._canvas_img_id is None:
                self._canvas_img_id = self._canvas.create_image(
                    ox, oy, anchor="nw", image=self._tk_img_ref)
            else:
                self._canvas.coords(self._canvas_img_id, ox, oy)
                self._canvas.itemconfig(self._canvas_img_id, image=self._tk_img_ref)
        finally:
            self._pending_draw = False

    # ── Stats poll ────────────────────────────────────────────────────────────
    def _poll_stats(self):
        self._qr_count_var.set(str(self._qr_count))
        self._ok_count_var.set(str(self._ok_count))
        self._ng_count_var.set(str(self._ng_count))
        self._total_var.set(str(self._total_scanned))
        self._total_ok_var.set(str(self._total_ok))
        self._total_ng_var.set(str(self._total_ng))
        self._fps_var.set(f"{self._fps_val:.1f}")
        self._cam_status_var.set(self._camera_status)
        self._model_status_var.set(self._model_status)
        self._arduino_stat_var.set(self._arduino_status_var.get())
        self._lock_stat_var.set("🔒 Còn NG" if self._motor_locked else "🔓 OK")
        self.after(200, self._poll_stats)

    # ── Chụp ảnh train ────────────────────────────────────────────────────────
    def _capture_training_image(self):
        if not self._det_running:
            return
        frame = self._camera.read()
        if frame is None:
            self._capture_status_var.set("Chưa có frame, thử lại sau giây lát.")
            return
        ensure_dirs()
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"cap_{ts}_{ms:03d}.jpg"
        path = os.path.join("train_images", filename)
        cv2.imwrite(path, frame)

        total = 0
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            total += len(glob.glob(os.path.join("train_images", ext)))
        self._capture_status_var.set(f"✓ Đã lưu {filename}\nTổng: {total} ảnh trong train_images/")
        self._capture_btn.configure(text="✅  Đã lưu!")
        self.after(500, lambda: self._capture_btn.configure(text="📷  Chụp ảnh Train"))

    # ── Mở thư mục kết quả ───────────────────────────────────────────────────
    def _open_ok_folder(self):
        self._open_folder(self._result_logger.ok_dir)

    def _open_ng_folder(self):
        self._open_folder(self._result_logger.ng_dir)

    def _open_folder(self, path: str):
        os.makedirs(path, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không mở được thư mục:\n{e}")

    # ── Sub-windows ───────────────────────────────────────────────────────────
    def _open_annotation(self):
        AnnotationWindow(self).grab_set()

    def _open_training(self):
        TrainingWindow(self, on_trained_callback=self._on_model_trained).grab_set()

    def _on_model_trained(self, model_path):
        self._config["model_path"] = model_path
        save_config(self._config)
        self._detector.update_model(model_path)
        self._model_status = "Loaded ✓ (new)"
        self._device_var.set("GPU" if self._detector._device.startswith("cuda") else "CPU")

    def _open_settings(self):
        SettingsDialog(self, self._config, on_save=self._apply_settings)

    def _apply_settings(self, new_config):
        self._config = new_config
        save_config(new_config)
        self._detector.update_threshold(new_config["confidence_threshold"])
        self._detector.update_imgsz(new_config.get("infer_imgsz", 320))
        self._skip_frames = new_config.get("skip_frames", 1)
        self._tracker.update_confirm_distance(new_config.get("confirm_distance_px", 150))
        self._conf_disp_var.set(f"{new_config['confidence_threshold']:.0%}")
        self._result_logger.set_output_dir(new_config.get("result_dir", "results"))
        self._init_arduino()  # Reconnect nếu port / enabled thay đổi

    def on_close(self):
        self._det_running = False
        self._camera.stop()
        if self._arduino is not None:
            try:
                self._arduino.close()
            except Exception:
                pass
        self.destroy()