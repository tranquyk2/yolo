# qr_tracker.py v5
# Global greedy IoU matching + center-distance fallback cho bang chuyen nhanh
import time
import math
from typing import Callable, Optional, List, Dict, Any, Tuple


def _iou(a: Dict, b: Dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / (area_a + area_b - inter)


def _center_dist(tr_box: Dict, dcx: float, dcy: float) -> float:
    cx = (tr_box["x1"] + tr_box["x2"]) / 2
    cy = (tr_box["y1"] + tr_box["y2"]) / 2
    return math.hypot(cx - dcx, cy - dcy)


class Track:
    def __init__(self, track_id: int, det: Dict[str, Any],
                 confirm_distance_px: float = 150.0, min_hits_floor: int = 2):
        self.id = track_id
        self.x1 = det["x1"]
        self.y1 = det["y1"]
        self.x2 = det["x2"]
        self.y2 = det["y2"]
        self.misses = 0
        self.hit_count = 1
        # Quãng đường (px) cần đi được trước khi "chốt" NG — TỰ ĐỘNG thích
        # nghi với tốc độ băng chuyền: băng nhanh -> QR di chuyển nhiều px
        # mỗi lần detect -> đủ quãng đường sau ít lần hit hơn. Băng chậm ->
        # ngược lại, tự động cần nhiều lần hit hơn. Không cần chỉnh tay khi
        # tốc độ đổi (ví dụ cuộn tem to dần lên).
        self.confirm_distance_px = confirm_distance_px
        # Số hit tối thiểu để chặn trường hợp 1-2 frame nhiễu/jump lớn do
        # distance-fallback matching bị tính nhầm là "đã đi đủ xa".
        self.min_hits_floor = min_hits_floor
        self.ever_ok = (det.get("status") == "OK")
        self.best_text = det.get("text", "") if self.ever_ok else ""
        self.first_seen = time.time()
        # Vị trí tâm box lúc mới xuất hiện — dùng để đo quãng đường đã đi
        self.first_cx = (self.x1 + self.x2) / 2
        self.first_cy = (self.y1 + self.y2) / 2

    def as_box(self) -> Dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def travel_distance(self) -> float:
        """Quãng đường (px) đã di chuyển kể từ lúc track được tạo."""
        return math.hypot(self.cx - self.first_cx, self.cy - self.first_cy)

    def update(self, det: Dict[str, Any]):
        self.x1, self.y1, self.x2, self.y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        self.misses = 0
        self.hit_count += 1
        if det.get("status") == "OK" and not self.ever_ok:
            self.ever_ok = True
            self.best_text = det.get("text", "")

    @property
    def is_mature(self) -> bool:
        """Track được coi là 'đã đủ tin cậy' khi vừa đủ số hit tối thiểu
        (chống nhiễu 1 frame), VỪA đã di chuyển đủ quãng đường yêu cầu
        (tự động thích nghi tốc độ băng chuyền thực tế)."""
        return (self.hit_count >= self.min_hits_floor
                and self.travel_distance >= self.confirm_distance_px)

    @property
    def status(self) -> str:
        return "OK" if self.ever_ok else "NG"

    @property
    def display_status(self) -> str:
        if self.ever_ok:
            return "OK"
        if not self.is_mature:
            return "..."
        return "NG"


class QRTracker:
    def __init__(
        self,
        min_iou: float = 0.15,
        max_misses: int = 15,
        confirm_distance_px: float = 150.0,
        min_hits_floor: int = 2,
        on_finalize: Optional[Callable[["Track"], None]] = None,
        max_center_dist: float = 200.0,
        max_distance: float = 0,
    ):
        self.min_iou = min_iou
        self.max_center_dist = max_center_dist
        self.max_misses = max_misses
        self.confirm_distance_px = confirm_distance_px
        self.min_hits_floor = min_hits_floor
        self.on_finalize = on_finalize
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Buoc 1: IoU matching (global greedy, uu tien truoc)
        candidate_pairs: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for tid, tr in self.tracks.items():
                score = _iou(det, tr.as_box())
                if score > self.min_iou:
                    candidate_pairs.append((score, det_idx, tid))

        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        det_to_track: Dict[int, int] = {}
        used_tracks: set = set()
        for score, det_idx, tid in candidate_pairs:
            if det_idx in det_to_track or tid in used_tracks:
                continue
            det_to_track[det_idx] = tid
            used_tracks.add(tid)

        # Buoc 2: Distance fallback cho bang chuyen nhanh
        # Khi QR di chuyen xa giua 2 lan detect (IoU=0), van nhan ra cung 1 vat
        # bang khoang cach tam box thay vi IoU.
        if self.max_center_dist > 0:
            unmatched_dets = [i for i in range(len(detections))
                              if i not in det_to_track]
            free_tracks = {tid: tr for tid, tr in self.tracks.items()
                           if tid not in used_tracks}

            dist_pairs: List[Tuple[float, int, int]] = []
            for det_idx in unmatched_dets:
                det = detections[det_idx]
                dcx = (det["x1"] + det["x2"]) / 2
                dcy = (det["y1"] + det["y2"]) / 2
                for tid, tr in free_tracks.items():
                    d = _center_dist(tr.as_box(), dcx, dcy)
                    if d <= self.max_center_dist:
                        dist_pairs.append((d, det_idx, tid))

            dist_pairs.sort(key=lambda x: x[0])
            for dist, det_idx, tid in dist_pairs:
                if det_idx in det_to_track or tid in used_tracks:
                    continue
                det_to_track[det_idx] = tid
                used_tracks.add(tid)

        # Cap nhat track da match / tao track moi cho detection chua match
        matched_track_ids: set = set()
        for det_idx, det in enumerate(detections):
            matched_tid = det_to_track.get(det_idx)

            if matched_tid is not None:
                self.tracks[matched_tid].update(det)
                tid = matched_tid
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, det, self.confirm_distance_px, self.min_hits_floor)

            matched_track_ids.add(tid)
            tr = self.tracks[tid]
            det["track_id"] = tid
            det["status"] = tr.display_status

        # Tang miss / chot track khong xuat hien
        gone = []
        for tid, tr in self.tracks.items():
            if tid not in matched_track_ids:
                tr.misses += 1
                if tr.misses > self.max_misses:
                    gone.append(tid)

        for tid in gone:
            tr = self.tracks.pop(tid)
            if tr.ever_ok:
                if self.on_finalize:
                    self.on_finalize(tr)
            elif tr.is_mature:
                if self.on_finalize:
                    self.on_finalize(tr)

        return detections

    def active_count(self) -> int:
        return len(self.tracks)

    def update_confirm_distance(self, value: float):
        """Đổi quãng đường (px) cần đi được trước khi 1 track thoát trạng
        thái ô xám '...' và chốt thành NG (OK vẫn chốt ngay khi decode
        thành công lần đầu, không phụ thuộc giá trị này).
        Chỉ áp dụng cho track MỚI được tạo sau lệnh này — track đang
        active giữa chừng vẫn dùng giá trị cũ """
        self.confirm_distance_px = max(1.0, float(value))