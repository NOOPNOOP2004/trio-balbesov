# people_counter_robust.py
# ==========================================================
# УЛУЧШЕННЫЙ СЧЁТЧИК ЛЮДЕЙ v2
# Исправления относительно v1:
#
# ГЛАВНАЯ ПРОБЛЕМА v1 — blaze_face_short_range.tflite
#   Модель рассчитана на расстояние ~2м (режим «селфи»).
#   Люди на заднем плане имеют маленькие лица → модель их не видит.
#   Без лица YOLO-тело не создаёт человека → баг с фоновыми людьми.
#
# ЧТО ИСПРАВЛЕНО:
#   1. Full-range модель (primary) + short-range (fallback)
#   2. Мультимасштабное обнаружение лиц (0.6x / 1.0x / 1.5x)
#   3. Создание человека только по телу (body-only fallback)
#      — для стабильных YOLO-треков без лица ≥ BODY_ONLY_MIN_FRAMES кадров
#   4. CLAHE + опциональный денойз для дешёвых веб-камер
#   5. NMS для слияния дублей при мультимасштабе
#   6. Адаптивный порог расстояния: маленькое лицо = далёкий человек
#   7. DEBUG-режим: показывает тела и диагностику
#
# Управление:
#   Q / ESC  — выход
#   R        — сброс
#   S        — скриншот
#   D        — переключить DEBUG overlay
# ==========================================================

import cv2
import csv
import time
import math
import argparse
import platform
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


# ==========================================================
# НАСТРОЙКИ
# ==========================================================

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

LOG_FOLDER        = Path("logs")
MODEL_FOLDER      = Path("models")
SCREENSHOT_FOLDER = Path("screenshots")

CSV_FILE = LOG_FOLDER / "people_events_v2.csv"

# Модели MediaPipe
FACE_FULL_MODEL_FILE  = MODEL_FOLDER / "blaze_face_full_range.tflite"
FACE_SHORT_MODEL_FILE = MODEL_FOLDER / "blaze_face_short_range.tflite"

FACE_FULL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_full_range/float16/1/"
    "blaze_face_full_range.tflite"
)
FACE_SHORT_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)

DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_TRACKER    = "botsort.yaml"

FACE_CONFIDENCE      = 0.50   # ниже чем в v1 — чтобы ловить далёкие лица
YOLO_CONFIDENCE      = 0.30
YOLO_IOU             = 0.45

FACE_CONFIRM_SECONDS = 0.4
FACE_CONFIRM_FRAMES  = 2

DEFAULT_HOLD_SECONDS    = 8.0
FORGET_AFTER_SECONDS    = 30.0
FACE_REID_SECONDS       = 15.0
BODY_REID_SECONDS       = 10.0

# v2: минимальная площадь лица снижена — ловим маленькие дальние лица
MIN_FACE_AREA_RATIO = 0.0008   # было 0.0015

# v2: body-only fallback — сколько кадров YOLO должен стабильно видеть тело
# прежде чем создать человека без лица
BODY_ONLY_MIN_FRAMES = 6       # ~0.2 сек при 30fps
BODY_ONLY_MIN_CONF   = 0.45

# v2: мультимасштаб — коэффициенты изменения размера
MULTISCALE_FACTORS = [0.6, 1.0, 1.5]

# v2: NMS порог для слияния дублей от мультимасштаба
MULTISCALE_NMS_IOU  = 0.35


# ==========================================================
# ВСПОМОГАТЕЛЬНЫЕ: ВРЕМЯ, ПАПКИ, CSV
# ==========================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def prepare_folders_and_csv():
    LOG_FOLDER.mkdir(exist_ok=True)
    MODEL_FOLDER.mkdir(exist_ok=True)
    SCREENSHOT_FOLDER.mkdir(exist_ok=True)

    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            "datetime", "event", "people_now", "total_people",
            "active_person_ids", "visible_face_count",
            "visible_body_count", "comment"
        ])


def append_csv(event, people_now, total_people, active_ids,
               face_count, body_count, comment):
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            now_str(), event, people_now, total_people,
            ",".join(map(str, active_ids)),
            face_count, body_count, comment
        ])
        f.flush()


def save_screenshot(frame):
    SCREENSHOT_FOLDER.mkdir(exist_ok=True)
    filename = datetime.now().strftime("screenshot_%Y-%m-%d_%H-%M-%S.jpg")
    path = SCREENSHOT_FOLDER / filename
    cv2.imwrite(str(path), frame)
    print(f"[INFO] Скриншот: {path}")


# ==========================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================================

def download_model(url: str, path: Path):
    if path.exists():
        print(f"[INFO] Найдена модель: {path.name}")
        return
    print(f"[INFO] Скачиваю {path.name} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] Готово: {path.name}")
    except Exception as e:
        print(f"[ERROR] Не удалось скачать {path.name}: {e}")
        raise


# ==========================================================
# ГЕОМЕТРИЯ
# ==========================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def box_area(box):
    w, h = box_size(box)
    return w * h


def center_distance(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.dist((ax, ay), (bx, by))


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter    = iw * ih
    union    = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_box(box, factor, fw, fh):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (
        max(0,    int(x1 - w * factor)),
        max(0,    int(y1 - h * factor)),
        min(fw-1, int(x2 + w * factor)),
        min(fh-1, int(y2 + h * factor)),
    )


def point_inside_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def face_inside_body(face_box, body_box, fw, fh):
    exp = expand_box(body_box, 0.15, fw, fh)
    cx, cy = box_center(face_box)
    return point_inside_box((cx, cy), exp) or box_iou(face_box, exp) > 0.01


# v2: адаптивный радиус поиска по размеру лица
def adaptive_max_distance(box_a, box_b, base_factor=1.4):
    aw, ah = box_size(box_a)
    bw, bh = box_size(box_b)
    ref = max(aw, ah, bw, bh)
    # маленькое лицо = далёкий человек = ищем шире
    small_bonus = 1.0 + max(0.0, (60 - ref) / 60.0) * 0.8
    return ref * base_factor * small_bonus


def nms_faces(faces, iou_thresh=MULTISCALE_NMS_IOU):
    """NMS для слияния дублей при мультимасштабном обнаружении."""
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: f["score"], reverse=True)
    keep = []
    suppressed = set()
    for i, f in enumerate(faces):
        if i in suppressed:
            continue
        keep.append(f)
        for j in range(i + 1, len(faces)):
            if j in suppressed:
                continue
            if box_iou(f["box"], faces[j]["box"]) >= iou_thresh:
                suppressed.add(j)
    return keep


def body_reid_candidate(old_box, new_box):
    old_area = box_area(old_box)
    new_area = box_area(new_box)
    if old_area <= 0 or new_area <= 0:
        return False
    ratio = new_area / float(old_area)
    if ratio < 0.30 or ratio > 3.2:
        return False
    iou  = box_iou(old_box, new_box)
    dist = center_distance(old_box, new_box)
    ow, oh = box_size(old_box)
    nw, nh = box_size(new_box)
    ref  = max(ow, oh, nw, nh)
    return iou >= 0.12 or dist <= ref * 0.45


def face_reid_candidate(old_face, new_face):
    iou  = box_iou(old_face, new_face)
    dist = center_distance(old_face, new_face)
    max_d = adaptive_max_distance(old_face, new_face)
    return iou >= 0.10 or dist <= max_d


# ==========================================================
# ПРЕДОБРАБОТКА КАДРА (v2)
# Помогает дешёвым камерам: повышает контраст + опционально шумоподавление
# ==========================================================

def preprocess_frame(frame: np.ndarray, denoise: bool = False) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) на Y-канале YUV.
    Значительно улучшает детектирование лиц на дешёвых камерах.
    """
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    if denoise:
        # Быстрый лёгкий блюр вместо тяжёлого fastNlMeans
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

    return enhanced


# ==========================================================
# ДЕТЕКТОР ЛИЦ С МУЛЬТИМАСШТАБОМ (v2)
# ==========================================================

class MultiScaleFaceDetector:
    """
    Запускает MediaPipe на нескольких масштабах,
    потом объединяет все детекции через NMS.
    Это позволяет находить лица людей на заднем плане.
    """
    def __init__(self, model_path: Path, min_confidence: float):
        base_opts = python.BaseOptions(model_asset_path=str(model_path))
        opts = vision.FaceDetectorOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
            min_suppression_threshold=0.25,
        )
        self.detector = vision.FaceDetector.create_from_options(opts)
        self.last_ts_ms = 0

    def _detect_single(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Один прогон детектора на переданном кадре."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect_for_video(mp_img, timestamp_ms)
        faces = []

        if not result.detections:
            return faces, w, h

        for det in result.detections:
            bb = det.bounding_box
            x1 = max(0, min(w-1, int(bb.origin_x)))
            y1 = max(0, min(h-1, int(bb.origin_y)))
            x2 = max(0, min(w-1, int(bb.origin_x + bb.width)))
            y2 = max(0, min(h-1, int(bb.origin_y + bb.height)))

            if x2 <= x1 or y2 <= y1:
                continue

            area_ratio = box_area((x1, y1, x2, y2)) / float(w * h)
            if area_ratio < MIN_FACE_AREA_RATIO:
                continue

            score = float(det.categories[0].score) if det.categories else 0.0
            faces.append({"box": (x1, y1, x2, y2), "score": score})

        return faces, w, h

    def detect(self, frame_bgr: np.ndarray):
        """
        Мультимасштабное обнаружение:
        - scale < 1: уменьшаем кадр → быстрее, ловим ближних крупных
        - scale = 1: оригинал
        - scale > 1: увеличиваем → ловим мелкие дальние лица
        Все результаты переводятся в исходные координаты и сливаются через NMS.
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        all_faces = []

        ts_ms = int(time.time() * 1000)
        if ts_ms <= self.last_ts_ms:
            ts_ms = self.last_ts_ms + 1
        self.last_ts_ms = ts_ms

        for scale in MULTISCALE_FACTORS:
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            if new_w < 64 or new_h < 64:
                continue

            if abs(scale - 1.0) < 0.05:
                scaled = frame_bgr
            else:
                interp = cv2.INTER_LINEAR if scale < 1.0 else cv2.INTER_CUBIC
                scaled = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)

            # Инкрементируем timestamp для каждого масштаба
            ts_ms += 1
            self.last_ts_ms = ts_ms

            faces_scaled, sw, sh = self._detect_single(scaled, ts_ms)

            # Перевод координат в оригинальный масштаб
            for f in faces_scaled:
                x1, y1, x2, y2 = f["box"]
                ox1 = int(x1 / scale)
                oy1 = int(y1 / scale)
                ox2 = int(x2 / scale)
                oy2 = int(y2 / scale)
                # Клип по оригинальному размеру
                ox1 = max(0, min(orig_w-1, ox1))
                oy1 = max(0, min(orig_h-1, oy1))
                ox2 = max(0, min(orig_w-1, ox2))
                oy2 = max(0, min(orig_h-1, oy2))
                if ox2 > ox1 and oy2 > oy1:
                    all_faces.append({
                        "box":   (ox1, oy1, ox2, oy2),
                        "score": f["score"],
                        "scale": scale,  # для DEBUG
                    })

        # Слияние дублей через NMS
        merged = nms_faces(all_faces, MULTISCALE_NMS_IOU)
        return merged

    def close(self):
        self.detector.close()


# ==========================================================
# YOLO
# ==========================================================

def load_yolo_model(model_name: str):
    print(f"[INFO] Загружается YOLO: {model_name}")
    return YOLO(model_name)


def extract_yolo_tracks(result):
    tracks = []
    if result.boxes is None or result.boxes.id is None:
        return tracks

    xyxy    = result.boxes.xyxy.cpu().tolist()
    ids     = result.boxes.id.int().cpu().tolist()
    classes = result.boxes.cls.int().cpu().tolist()
    confs   = result.boxes.conf.cpu().tolist()

    for box, tid, cls, conf in zip(xyxy, ids, classes, confs):
        if int(cls) != 0:  # только class 0 = person
            continue
        x1, y1, x2, y2 = map(int, box)
        if x2 <= x1 or y2 <= y1:
            continue
        tracks.append({
            "yolo_id": int(tid),
            "box":     (x1, y1, x2, y2),
            "conf":    float(conf),
        })

    return tracks


# ==========================================================
# СОСТОЯНИЕ ЧЕЛОВЕКА
# ==========================================================

class PersonState:
    def __init__(self, pid, face_box, face_score, ts, body_only=False):
        self.person_id       = pid
        self.first_seen      = ts
        self.last_face_seen  = ts if not body_only else 0.0
        self.last_body_seen  = ts if body_only else 0.0
        self.face_hits       = 0 if body_only else 1
        self.body_hits       = 1 if body_only else 0
        self.confirmed       = False
        self.counted_total   = False
        self.last_face_box   = face_box        # None если body_only
        self.last_face_score = face_score
        self.last_body_box   = None
        self.last_body_conf  = 0.0
        self.yolo_ids        = set()
        self.body_only       = body_only       # v2: флаг создан без лица

    def update_face(self, face_box, face_score, ts):
        self.last_face_seen  = ts
        self.face_hits      += 1
        self.last_face_box   = face_box
        self.last_face_score = face_score
        self.body_only       = False  # появилось лицо → снимаем флаг

    def update_body(self, yolo_id, body_box, body_conf, ts):
        self.yolo_ids.add(yolo_id)
        self.last_body_seen  = ts
        self.body_hits      += 1
        self.last_body_box   = body_box
        self.last_body_conf  = body_conf

    def last_seen_any(self):
        return max(self.last_face_seen, self.last_body_seen)

    def should_confirm(self, current_time):
        if self.body_only:
            # Тело без лица подтверждается медленнее
            return self.body_hits >= BODY_ONLY_MIN_FRAMES
        if self.face_hits >= FACE_CONFIRM_FRAMES:
            return True
        return current_time - self.first_seen >= FACE_CONFIRM_SECONDS


# ==========================================================
# СЧЁТЧИК
# ==========================================================

class PeopleCounter:
    def __init__(self, hold_seconds: float):
        self.hold_seconds           = hold_seconds
        self.next_person_id         = 1
        self.people                 = {}        # pid → PersonState
        self.yolo_to_person         = {}        # yolo_id → pid
        self.total_people           = 0
        self.last_logged_people_now = 0
        self.last_logged_total      = 0
        # v2: трекер для body-only кандидатов
        # yolo_id → {"frames": int, "box": ..., "conf": float}
        self._body_candidates       = {}

    def reset(self):
        self.__init__(self.hold_seconds)

    # --------------------------------------------------
    # Вспомогательные поиски
    # --------------------------------------------------

    def _find_face_match(self, face_box, current_time, used_pids):
        best_pid, best_score = None, 999999.0
        for pid, p in self.people.items():
            if pid in used_pids:
                continue
            if current_time - p.last_face_seen > FACE_REID_SECONDS:
                continue
            if p.last_face_box is None:
                continue
            if not face_reid_candidate(p.last_face_box, face_box):
                continue
            score = center_distance(p.last_face_box, face_box) - box_iou(p.last_face_box, face_box) * 200
            if score < best_score:
                best_score, best_pid = score, pid
        return best_pid

    def _find_body_match(self, body_box, current_time, used_pids):
        best_pid, best_score = None, 999999.0
        for pid, p in self.people.items():
            if pid in used_pids:
                continue
            if not p.confirmed:
                continue
            if p.last_body_box is None:
                continue
            if current_time - p.last_body_seen > BODY_REID_SECONDS:
                continue
            if not body_reid_candidate(p.last_body_box, body_box):
                continue
            score = center_distance(p.last_body_box, body_box) - box_iou(p.last_body_box, body_box) * 150
            if score < best_score:
                best_score, best_pid = score, pid
        return best_pid

    def _find_best_body_for_face(self, face_box, tracks, used_yids, fw, fh):
        best_body, best_score = None, 999999.0
        for body in tracks:
            if body["yolo_id"] in used_yids:
                continue
            if not face_inside_body(face_box, body["box"], fw, fh):
                continue
            score = center_distance(face_box, body["box"]) - box_iou(face_box, body["box"]) * 300
            if score < best_score:
                best_score, best_body = score, body
        return best_body

    def _create_person(self, face_box, face_score, ts, body_only=False):
        pid = self.next_person_id
        self.next_person_id += 1
        self.people[pid] = PersonState(pid, face_box, face_score, ts, body_only)
        return pid

    # --------------------------------------------------
    # ГЛАВНЫЙ UPDATE
    # --------------------------------------------------

    def update(self, faces, body_tracks, current_time, fw, fh):
        used_pids  = set()
        used_yids  = set()

        faces_sorted = sorted(faces, key=lambda f: f["score"], reverse=True)

        # --- ШАГ 1: Лица создают / обновляют людей ---
        for face in faces_sorted:
            fb, fs = face["box"], face["score"]

            best_body = self._find_best_body_for_face(fb, body_tracks, used_yids, fw, fh)

            pid = None

            # Смотрим, есть ли уже человек с этим YOLO-треком
            if best_body is not None:
                yid = best_body["yolo_id"]
                mapped = self.yolo_to_person.get(yid)
                if mapped in self.people and mapped not in used_pids:
                    pid = mapped

            # Иначе ищем по позиции лица
            if pid is None:
                pid = self._find_face_match(fb, current_time, used_pids)

            # Если не нашли — создаём нового
            if pid is None:
                pid = self._create_person(fb, fs, current_time)
            else:
                self.people[pid].update_face(fb, fs, current_time)

            used_pids.add(pid)

            if best_body is not None:
                yid = best_body["yolo_id"]
                self.people[pid].update_body(yid, best_body["box"], best_body["conf"], current_time)
                self.yolo_to_person[yid] = pid
                used_yids.add(yid)
                # Очищаем кандидата на body-only если он был
                self._body_candidates.pop(yid, None)

        # --- ШАГ 2: Тела обновляют существующих ---
        for body in body_tracks:
            yid = body["yolo_id"]
            if yid in used_yids:
                continue

            bb, bc = body["box"], body["conf"]

            pid = self.yolo_to_person.get(yid)

            if pid in self.people:
                self.people[pid].update_body(yid, bb, bc, current_time)
                used_yids.add(yid)
                self._body_candidates.pop(yid, None)
                continue

            # Ищем совпадение по положению тела
            pid = self._find_body_match(bb, current_time, used_pids)

            if pid in self.people:
                self.people[pid].update_body(yid, bb, bc, current_time)
                self.yolo_to_person[yid] = pid
                used_yids.add(yid)
                self._body_candidates.pop(yid, None)
                continue

            # --- v2: body-only fallback ---
            # Если ни один человек не совпал → накапливаем кандидата
            if bc >= BODY_ONLY_MIN_CONF:
                if yid not in self._body_candidates:
                    self._body_candidates[yid] = {"frames": 1, "box": bb, "conf": bc}
                else:
                    cand = self._body_candidates[yid]
                    cand["frames"] += 1
                    cand["box"]     = bb
                    cand["conf"]    = bc

                    if cand["frames"] >= BODY_ONLY_MIN_FRAMES:
                        # Достаточно кадров → создаём человека без лица
                        pid = self._create_person(
                            face_box   = None,
                            face_score = 0.0,
                            ts         = current_time,
                            body_only  = True,
                        )
                        self.people[pid].update_body(yid, bb, bc, current_time)
                        self.yolo_to_person[yid] = pid
                        used_yids.add(yid)
                        used_pids.add(pid)
                        del self._body_candidates[yid]
                        print(f"[v2] Body-only person #{pid} создан (YOLO id={yid})")
            else:
                self._body_candidates.pop(yid, None)

        # Удаляем body_candidates для пропавших треков
        active_yids = {b["yolo_id"] for b in body_tracks}
        stale_cands = [yid for yid in self._body_candidates if yid not in active_yids]
        for yid in stale_cands:
            del self._body_candidates[yid]

        # --- ШАГ 3: Подтверждение и total ---
        for pid, p in self.people.items():
            if not p.confirmed and p.should_confirm(current_time):
                p.confirmed = True
            if p.confirmed and not p.counted_total:
                p.counted_total = True
                self.total_people += 1

        # --- ШАГ 4: Удаление давно пропавших ---
        to_del = [
            pid for pid, p in self.people.items()
            if current_time - p.last_seen_any() > FORGET_AFTER_SECONDS
        ]
        for pid in to_del:
            p = self.people.pop(pid, None)
            if p:
                for yid in p.yolo_ids:
                    self.yolo_to_person.pop(yid, None)

        # --- ШАГ 5: Активные сейчас ---
        active_pids = sorted([
            pid for pid, p in self.people.items()
            if p.confirmed and current_time - p.last_seen_any() <= self.hold_seconds
        ])

        return {
            "people_now":         len(active_pids),
            "total_people":       self.total_people,
            "active_person_ids":  active_pids,
            "visible_face_count": len(faces),
            "visible_body_count": len(body_tracks),
        }


# ==========================================================
# КАМЕРА
# ==========================================================

def open_camera(index: int):
    sys = platform.system().lower()
    if "windows" in sys:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    elif "linux" in sys:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# ==========================================================
# ОТРИСОВКА
# ==========================================================

def draw_interface(frame, faces, body_tracks, counter, stats, fps, debug_mode):
    h, w = frame.shape[:2]

    # Рамки лиц
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        score = face["score"]
        scale_info = face.get("scale", 1.0)

        # Цвет: зелёный для близких, синий для мультимасштаб-дальних
        color = (0, 200, 0) if scale_info >= 1.0 else (255, 160, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"FACE {score:.2f}"
        if debug_mode and abs(scale_info - 1.0) > 0.05:
            label += f" [s={scale_info:.1f}]"

        cv2.putText(frame, label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # DEBUG: рамки тел
    if debug_mode:
        for body in body_tracks:
            x1, y1, x2, y2 = body["box"]
            yid  = body["yolo_id"]
            conf = body["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 1)
            cv2.putText(frame, f"BODY id={yid} {conf:.2f}",
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        # Кандидаты body-only
        for yid, cand in counter._body_candidates.items():
            x1, y1, x2, y2 = cand["box"]
            frames = cand["frames"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"CAND {frames}/{BODY_ONLY_MIN_FRAMES}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Верхняя панель
    panel_h = 195
    cv2.rectangle(frame, (0, 0), (w, panel_h), (20, 20, 20), -1)

    cv2.putText(frame, "People Counter v2  [multi-scale | body fallback]",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.putText(frame, f"People now: {stats['people_now']}",
                (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.putText(frame, f"Total: {stats['total_people']}",
                (330, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.putText(frame, f"Faces: {stats['visible_face_count']}",
                (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.putText(frame, f"Bodies: {stats['visible_body_count']}",
                (170, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (350, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.putText(frame, f"Hold: {counter.hold_seconds:.1f}s",
                (470, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    debug_label = "DEBUG ON" if debug_mode else "debug off"
    debug_color = (0, 255, 255) if debug_mode else (100, 100, 100)
    cv2.putText(frame, f"[D] {debug_label}",
                (700, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, debug_color, 2)

    cv2.putText(
        frame,
        "Q/ESC exit | R reset | S screenshot | D debug",
        (20, 162),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1
    )

    return frame


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="People Counter v2")
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--yolo-model",type=str,   default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--tracker",   type=str,   default=DEFAULT_TRACKER)
    parser.add_argument("--face-conf", type=float, default=FACE_CONFIDENCE)
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONFIDENCE)
    parser.add_argument("--hold",      type=float, default=DEFAULT_HOLD_SECONDS)
    parser.add_argument("--denoise",   action="store_true",
                        help="Включить шумоподавление (медленнее, лучше для плохих камер)")
    parser.add_argument("--no-multiscale", action="store_true",
                        help="Отключить мультимасштабное детектирование (быстрее, хуже для фона)")
    parser.add_argument("--debug",     action="store_true",
                        help="Показать overlay тел и диагностику")
    parser.add_argument("--source",    type=str,   default=None,
                        help="Видеофайл вместо камеры (для тестирования в Docker)")
    args = parser.parse_args()

    print("==========================================")
    print(" People Counter v2  (улучшенная версия)")
    print(" multi-scale | body fallback | CLAHE")
    print("==========================================")

    prepare_folders_and_csv()

    # Скачиваем обе модели
    download_model(FACE_FULL_URL,  FACE_FULL_MODEL_FILE)
    download_model(FACE_SHORT_URL, FACE_SHORT_MODEL_FILE)

    # Пробуем full_range как основную; если нет — short_range
    face_model_path = FACE_FULL_MODEL_FILE
    if not face_model_path.exists():
        face_model_path = FACE_SHORT_MODEL_FILE
        print("[WARN] full_range модель недоступна, используется short_range")

    face_detector = MultiScaleFaceDetector(
        model_path     = face_model_path,
        min_confidence = args.face_conf,
    )

    # Если --no-multiscale: выставляем один масштаб
    global MULTISCALE_FACTORS
    if args.no_multiscale:
        MULTISCALE_FACTORS = [1.0]
        print("[INFO] Мультимасштаб отключён")

    yolo_model = load_yolo_model(args.yolo_model)
    counter    = PeopleCounter(hold_seconds=args.hold)

    # Источник видео: файл или камера
    if args.source:
        print(f"[INFO] Источник: файл {args.source}")
        cap = cv2.VideoCapture(args.source)
    else:
        cap = open_camera(args.camera)

    if cap is None or not cap.isOpened():
        print("[ERROR] Камера / файл не найдены.")
        return

    print(f"[INFO] Детектор: {face_model_path.name}")
    print(f"[INFO] Мультимасштаб: {MULTISCALE_FACTORS}")
    print(f"[INFO] CLAHE: всегда | Денойз: {args.denoise}")
    print(f"[INFO] Body-only fallback: {BODY_ONLY_MIN_FRAMES} кадров")
    print(f"[INFO] CSV: {CSV_FILE.resolve()}")

    append_csv("program_start", 0, 0, [], 0, 0, "v2 запущена")

    prev_time   = time.time()
    fps         = 0.0
    debug_mode  = args.debug
    last_stats  = {"people_now": 0, "total_people": 0,
                   "active_person_ids": [], "visible_face_count": 0,
                   "visible_body_count": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            # Для видеофайла — перемотка в начало
            if args.source:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[ERROR] Нет кадра.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        fh, fw = frame.shape[:2]

        # v2: предобработка
        proc_frame = preprocess_frame(frame, denoise=args.denoise)

        # Детектирование
        faces       = face_detector.detect(proc_frame)
        yolo_res    = yolo_model.track(
            proc_frame, persist=True, tracker=args.tracker,
            conf=args.yolo_conf, iou=YOLO_IOU,
            classes=[0], verbose=False
        )
        body_tracks = extract_yolo_tracks(yolo_res[0])

        current_time = time.time()
        stats        = counter.update(faces, body_tracks, current_time, fw, fh)

        # CSV при изменении
        pn_changed = stats["people_now"]   != counter.last_logged_people_now
        tot_changed = stats["total_people"] != counter.last_logged_total

        if pn_changed or tot_changed:
            ev = ("count_and_total_changed" if pn_changed and tot_changed
                  else ("count_changed" if pn_changed else "total_changed"))
            append_csv(
                ev,
                stats["people_now"],
                stats["total_people"],
                stats["active_person_ids"],
                stats["visible_face_count"],
                stats["visible_body_count"],
                f"now: {counter.last_logged_people_now}→{stats['people_now']}; "
                f"total: {counter.last_logged_total}→{stats['total_people']}"
            )
            counter.last_logged_people_now = stats["people_now"]
            counter.last_logged_total      = stats["total_people"]

        # FPS
        now   = time.time()
        delta = now - prev_time
        if delta > 0:
            fps = 1.0 / delta
        prev_time = now

        # Отрисовка (на оригинальном кадре, не на proc_frame)
        frame = draw_interface(frame, faces, body_tracks, counter, stats, fps, debug_mode)

        cv2.imshow("People Counter v2", frame)
        last_stats = stats

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("й"), 27):
            break
        elif key in (ord("r"), ord("к")):
            counter.reset()
            print("[INFO] Сброс.")
            append_csv("reset", 0, 0, [], 0, 0, "Сброс")
        elif key in (ord("s"), ord("ы")):
            save_screenshot(frame)
        elif key in (ord("d"), ord("в")):
            debug_mode = not debug_mode
            print(f"[INFO] DEBUG: {'ON' if debug_mode else 'OFF'}")

    append_csv("exit", last_stats["people_now"], last_stats["total_people"],
               last_stats["active_person_ids"], last_stats["visible_face_count"],
               last_stats["visible_body_count"], "Выход")

    face_detector.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n========== ГОТОВО ==========")
    print(f"Всего людей: {counter.total_people}")
    print(f"CSV: {CSV_FILE.resolve()}")


if __name__ == "__main__":
    main()