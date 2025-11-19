#!/usr/bin/env python3

"""

Features included:
- Creates required folders next to the EXE/.py (images, logs)
- Template normalization: supports many extensions and renames to image_001.png...
- Unicode-safe image IO (cv2.imdecode + tofile fallback)
- Win32 GDI fast screen capture and multi-frame median stabilization
- Multi-scale masked template matching with local-max suppression and NMS
- ORB + homography fallback
- Drawing result_image.png with edge-safe labels and bounding boxes
- Overwrites logs each run, writes start and finish timestamps only in headers
- CLI flags for runtime behavior (delete_found, produce_result_only_when_found)

"""

from __future__ import annotations
import os
import sys
import time
import math
import shutil
import argparse
import traceback
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from ctypes import wintypes
import ctypes

import numpy as np
import cv2

# ----------------------------
# PyInstaller-safe base paths
# ----------------------------
if getattr(sys, "frozen", False):
    # Running as PyInstaller EXE
    REAL_DIR = os.path.dirname(sys.executable)
    BASE_DIR = getattr(sys, "_MEIPASS", REAL_DIR)
else:
    REAL_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = REAL_DIR

# ----------------------------
# Configuration
# ----------------------------
IMAGE_DIR = os.path.join(REAL_DIR, "images")
LOGS_DIR = os.path.join(REAL_DIR, "logs")

LOG_PATH = os.path.join(LOGS_DIR, "ImageFinder.log")
COORDS_PATH = os.path.join(LOGS_DIR, "coords.log")
AHK_LOG_PATH = os.path.join(LOGS_DIR, "AhkStarter.log")
RESULT_IMAGE_PATH = os.path.join(LOGS_DIR, "result_image.png")

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Stabilization / capture
ULTRA_FRAMES = 6
FRAME_DELAY = 0.03

# Scale search
MIN_SCALE = 0.05
MAX_SCALE = 4.0
SCALE_STEPS = 36

# Thresholds
TM_THRESHOLD_SMALL = 0.985
TM_THRESHOLD_LARGE = 0.92
ORB_MIN_GOOD = 18
ORB_RATIO = 0.72

LOCAL_MAX_NEIGHBORHOOD = 9
IOU_MERGE_THRESHOLD = 0.35

TEMPLATE_PREFIX = "image_"
TEMPLATE_ZEROPAD = 3

MIN_BBOX_AREA_FRAC = 0.5

# Control defaults
DEFAULT_DELETE_FOUND = True
DEFAULT_KEEP_AHK_LOG = False
DEFAULT_RESULT_ONLY_WHEN_FOUND = True

# ----------------------------
# Utilities: time + safe IO
# ----------------------------

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_write_overwrite(path: str, text: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception:
        pass


def safe_append(path: str, text: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text)
    except Exception:
        pass

# Unicode-safe image read/write

def imread_unicode(path: str, flags=cv2.IMREAD_UNCHANGED):
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, flags)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1] or '.png'
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    try:
        cv2.imwrite(path, img)
        return True
    except Exception:
        return False

# ----------------------------
# Win32 GDI capture (fast)
# ----------------------------
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]

class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]


def grab_screen_gdi() -> np.ndarray:
    hdesktop = user32.GetDesktopWindow()
    hdc = user32.GetWindowDC(hdesktop)
    memdc = gdi32.CreateCompatibleDC(hdc)

    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)

    bmp = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(memdc, bmp)
    SRCCOPY = 0x00CC0020
    gdi32.BitBlt(memdc, 0, 0, width, height, hdc, 0, 0, SRCCOPY)

    bmi = _BITMAPINFO()
    ctypes.memset(ctypes.byref(bmi), 0, ctypes.sizeof(bmi))
    bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0

    buf_size = width * height * 4
    buffer = ctypes.create_string_buffer(buf_size)
    lines = gdi32.GetDIBits(memdc, bmp, 0, height, buffer, ctypes.byref(bmi), 0)

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(hdesktop, hdc)

    if lines != height:
        return np.zeros((height, width, 3), dtype=np.uint8)

    arr = np.frombuffer(buffer, dtype=np.uint8)
    arr = arr.reshape((height, width, 4))
    bgr = arr[:, :, :3].copy()
    return bgr


def capture_stabilized(frames: int = ULTRA_FRAMES, delay: float = FRAME_DELAY) -> np.ndarray:
    imgs = []
    for _ in range(max(1, frames)):
        try:
            imgs.append(grab_screen_gdi())
        except Exception:
            imgs.append(np.zeros((user32.GetSystemMetrics(1), user32.GetSystemMetrics(0), 3), dtype=np.uint8))
        time.sleep(delay)
    stack = np.stack(imgs, axis=3)
    median = np.median(stack, axis=3).astype(np.uint8)
    return median

# ----------------------------
# Helpers: IoU / NMS
# ----------------------------

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(boxes_scores, iou_thresh=IOU_MERGE_THRESHOLD):
    if not boxes_scores:
        return []
    boxes_scores = sorted(boxes_scores, key=lambda x: x[1], reverse=True)
    picked = []
    while boxes_scores:
        curr = boxes_scores.pop(0)
        picked.append(curr)
        boxes_scores = [b for b in boxes_scores if iou(curr[0], b[0]) <= iou_thresh]
    return picked

# ----------------------------
# ORB fallback
# ----------------------------

def orb_homography_find(screen_gray: np.ndarray, tpl_gray: np.ndarray, tpl_mask: Optional[np.ndarray]) -> Optional[List[List[int]]]:
    try:
        orb = cv2.ORB_create(4000)
        tpl_proc = tpl_gray.copy()
        if tpl_mask is not None:
            tpl_proc = cv2.bitwise_and(tpl_proc, tpl_proc, mask=tpl_mask)
        kp1, des1 = orb.detectAndCompute(tpl_proc, None)
        kp2, des2 = orb.detectAndCompute(screen_gray, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) != 2: continue
            m, n = m_n
            if m.distance < ORB_RATIO * n.distance:
                good.append(m)
        if len(good) < ORB_MIN_GOOD:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None
        h, w = tpl_gray.shape
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        projected = cv2.perspectiveTransform(corners, H)
        pts = projected.reshape(4,2).astype(int).tolist()
        bbox = [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        if area < max(4, w*h*MIN_BBOX_AREA_FRAC):
            return None
        return pts
    except Exception:
        return None

# ----------------------------
# Multiscale masked template matching (returns list of quads)
# ----------------------------

def multiscale_find_all(screen_gray: np.ndarray, tpl_gray: np.ndarray, tpl_mask: Optional[np.ndarray], tpl_w:int, tpl_h:int) -> List[List[List[int]]]:
    Hs, Ws = screen_gray.shape
    scales = np.geomspace(MIN_SCALE, MAX_SCALE, num=SCALE_STEPS)
    raw = []
    tpl_area = tpl_w * tpl_h
    tm_threshold = TM_THRESHOLD_SMALL if tpl_area < 1000 else TM_THRESHOLD_LARGE

    for scale in scales:
        th = max(1, int(tpl_h * scale))
        tw = max(1, int(tpl_w * scale))
        if th > Hs or tw > Ws:
            continue
        try:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(tpl_gray, (tw, th), interpolation=interp)
            if tpl_mask is not None:
                resized_mask = cv2.resize(tpl_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                if np.any(resized_mask > 0):
                    mean_val = int(np.mean(resized[resized_mask > 0]))
                    tmp = resized.copy()
                    tmp[resized_mask == 0] = mean_val
                    res = cv2.matchTemplate(screen_gray, tmp, cv2.TM_CCOEFF_NORMED)
                else:
                    res = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
            else:
                res = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
            if res is None:
                continue
            kernel = np.ones((LOCAL_MAX_NEIGHBORHOOD, LOCAL_MAX_NEIGHBORHOOD), np.uint8)
            dil = cv2.dilate(res, kernel)
            local_max = (res >= dil) & (res >= min(tm_threshold, 0.99))
            ys, xs = np.where(local_max)
            for (y, x) in zip(ys, xs):
                score = float(res[y, x])
                x1, y1, x2, y2 = int(x), int(y), int(x + tw), int(y + th)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(Ws - 1, x2), min(Hs - 1, y2)
                raw.append(((x1, y1, x2, y2), score, float(scale)))
        except Exception:
            continue

    boxes_scores = [(b, s, meta) for (b, s, meta) in raw]
    picked = nms(boxes_scores, iou_thresh=IOU_MERGE_THRESHOLD)
    results = []
    for box, score, scale in picked:
        x1,y1,x2,y2 = box
        w = max(1, x2-x1); h = max(1, y2-y1)
        bbox_area = w*h
        min_area = max(4, tpl_w * tpl_h * MIN_BBOX_AREA_FRAC * (scale if scale>0 else 1.0))
        if bbox_area < min_area:
            continue
        pts = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        results.append(pts)
    return results

# ----------------------------
# Draw helpers (edge-safe)
# ----------------------------
class DrawHelpers:
    @staticmethod
    def clamp_rect(x1,y1,x2,y2,W,H):
        x1_c = max(0, min(W-1, x1))
        y1_c = max(0, min(H-1, y1))
        x2_c = max(0, min(W-1, x2))
        y2_c = max(0, min(H-1, y2))
        return x1_c, y1_c, x2_c, y2_c

    @staticmethod
    def place_label(x, y, text_w, text_h, W, H, pad=6):
        tx = x
        ty = y - pad
        top = ty - text_h - pad
        left = tx - pad
        right = tx + text_w + pad
        if left < 0:
            tx += -left
        if right > W:
            tx -= (right - W)
        if top < 0:
            ty = y + text_h + pad + 2
            if ty + pad > H:
                ty = H - pad
        tx = max(2, min(tx, W - text_w - 2))
        ty = max(text_h + 2, min(ty, H - 2))
        return int(tx), int(ty)

# ----------------------------
# ImageFinder class (API)
# ----------------------------
class ImageFinder:
    def __init__(self,
                 image_dir: str = IMAGE_DIR,
                 logs_dir: str = LOGS_DIR,
                 delete_found: bool = DEFAULT_DELETE_FOUND,
                 keep_ahk_log: bool = DEFAULT_KEEP_AHK_LOG,
                 result_only_when_found: bool = DEFAULT_RESULT_ONLY_WHEN_FOUND):
        self.image_dir = image_dir
        self.logs_dir = logs_dir
        self.log_path = os.path.join(self.logs_dir, os.path.basename(LOG_PATH))
        self.coords_path = os.path.join(self.logs_dir, os.path.basename(COORDS_PATH))
        self.ahk_log_path = os.path.join(self.logs_dir, os.path.basename(AHK_LOG_PATH))
        self.result_path = os.path.join(self.logs_dir, os.path.basename(RESULT_IMAGE_PATH))
        self.delete_found = delete_found
        self.keep_ahk_log = keep_ahk_log
        self.result_only_when_found = result_only_when_found
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _init_logs(self):
        safe_write_overwrite(self.log_path, f"[{now_ts()}] ImageFinder run started\n")
        safe_write_overwrite(self.coords_path, f"[{now_ts()}] Coords summary\n\n")
        if self.keep_ahk_log:
            safe_write_overwrite(self.ahk_log_path, f"[{now_ts()}] AhkStarter placeholder log\n")

    def _log(self, msg: str):
        safe_append(self.log_path, f"[{now_ts()}] {msg}\n")

    def _finalize_logs(self):
        safe_append(self.log_path, f"[{now_ts()}] Run finished\n")
        safe_append(self.coords_path, f"[{now_ts()}] Coords end\n")
        if self.keep_ahk_log:
            safe_append(self.ahk_log_path, f"[{now_ts()}] AhkFinished\n")

    # Standardize templates (rename to image_001.png ...)
    def standardize_templates(self) -> List[str]:
        files = sorted([fn for fn in os.listdir(self.image_dir) if fn.lower().endswith(SUPPORTED_EXT)])
        if not files:
            return []
        standardized = []
        idx = 1
        for original in files:
            src = os.path.join(self.image_dir, original)
            target_name = f"{TEMPLATE_PREFIX}{idx:0{TEMPLATE_ZEROPAD}d}.png"
            dst = os.path.join(self.image_dir, target_name)
            img = imread_unicode(src)
            if img is None:
                try:
                    os.remove(src)
                except Exception:
                    pass
                idx += 1
                continue
            try:
                if imwrite_unicode(dst, img):
                    standardized.append(target_name)
                    if os.path.abspath(src) != os.path.abspath(dst):
                        try:
                            os.remove(src)
                        except Exception:
                            pass
                else:
                    # try fallback write using cv2.imwrite
                    try:
                        cv2.imwrite(dst, img)
                        standardized.append(target_name)
                        if os.path.abspath(src) != os.path.abspath(dst):
                            try: os.remove(src)
                            except: pass
                    except Exception:
                        pass
            except Exception:
                pass
            idx += 1
        return standardized

    def _read_template(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img = imread_unicode(path)
        if img is None:
            return None, None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None
        if img.shape[2] == 4:
            bgr = img[:, :, :3].copy()
            alpha = img[:, :, 3]
            mask = (alpha > 10).astype(np.uint8) * 255
            return bgr, mask
        return img[:, :, :3].copy(), None

    def _draw_and_save_result(self, screen_bgr: np.ndarray, detections: Dict[str, List[List[List[int]]]]):
        img = screen_bgr.copy()
        H, W = img.shape[:2]
        coords_out = {}
        for tpl, bboxes in detections.items():
            coords_out[tpl] = []
            for bbox in bboxes:
                try:
                    pts = np.array(bbox, dtype=np.int32)
                    x1 = int(pts[:,0].min()); y1 = int(pts[:,1].min())
                    x2 = int(pts[:,0].max()); y2 = int(pts[:,1].max())
                    x1, y1, x2, y2 = DrawHelpers.clamp_rect(x1,y1,x2,y2,W,H)
                    pts_clamped = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32)
                    cv2.polylines(img, [pts_clamped], True, (0,0,255), 3, cv2.LINE_AA)
                    label = tpl
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.7
                    thickness = 2
                    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
                    tx, ty = DrawHelpers.place_label(x1,y1,tw,th,W,H)
                    rect_tl = (max(0, tx-4), max(0, ty-th-4))
                    rect_br = (min(W-1, tx+tw+4), min(H-1, ty+4))
                    cv2.rectangle(img, rect_tl, rect_br, (0,0,255), -1)
                    cv2.putText(img, label, (tx, ty), font, scale, (255,255,255), thickness, cv2.LINE_AA)
                    coords_out[tpl].append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                except Exception:
                    continue
        # remove previous result
        try:
            if os.path.exists(self.result_path):
                os.remove(self.result_path)
        except Exception:
            pass
        imwrite_unicode(self.result_path, img)
        self._log(f"Saved result image: {self.result_path}")
        # coords
        lines = [f"[{now_ts()}] Coords summary", ""]
        for tpl, boxes in coords_out.items():
            lines.append(f"{tpl}:")
            if not boxes:
                lines.append("  NOT FOUND")
            else:
                for i,b in enumerate(boxes,1):
                    x1,y1 = b[0]
                    x2,y2 = b[2]
                    lines.append(f"  {i}. TopLeft=({x1},{y1})  BottomRight=({x2},{y2})")
            lines.append("")
        safe_write_overwrite(self.coords_path, "\n".join(lines)+"\n")

    def run(self) -> Dict[str, List[List[List[int]]]]:
        try:
            self._ensure_dirs()
            self._init_logs()
            self._log("Collecting templates")
            templates = self.standardize_templates()
            if not templates:
                self._log("No templates found in images/")
                # write minimal coords and finish
                safe_write_overwrite(self.coords_path, f"[{now_ts()}] Coords summary\n\nNo templates found\n")
                self._finalize_logs()
                return {}

            self._log(f"Templates: {', '.join(templates)}")

            # capture once
            self._log("Capturing stabilized screen")
            screen = capture_stabilized(ULTRA_FRAMES, FRAME_DELAY)
            screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

            detections: Dict[str, List[List[List[int]]]] = {}

            for tpl in templates:
                self._log(f"Processing template={tpl}")
                tpl_path = os.path.join(self.image_dir, tpl)
                tpl_bgr, tpl_mask = self._read_template(tpl_path)
                if tpl_bgr is None:
                    self._log(f"Failed to read template {tpl}")
                    detections[tpl] = []
                    continue
                tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
                tpl_h, tpl_w = tpl_gray.shape

                found = []
                try:
                    tmatches = multiscale_find_all(screen_gray, tpl_gray, tpl_mask, tpl_w, tpl_h)
                    if tmatches:
                        self._log(f"Template-match candidates for {tpl}: {len(tmatches)}")
                        found.extend(tmatches)
                    else:
                        self._log(f"No template-match candidates for {tpl}")
                except Exception as e:
                    self._log(f"Template matching error for {tpl}: {repr(e)}")

                if not found:
                    try:
                        orbres = orb_homography_find(screen_gray, tpl_gray, tpl_mask)
                        if orbres:
                            self._log(f"FOUND (orb) {tpl}")
                            found.append(orbres)
                        else:
                            self._log(f"ORB fallback none for {tpl}")
                    except Exception as e:
                        self._log(f"ORB error for {tpl}: {repr(e)}")

                # convert to boxes & NMS
                boxes_scores = []
                for poly in found:
                    x1 = min(p[0] for p in poly)
                    y1 = min(p[1] for p in poly)
                    x2 = max(p[0] for p in poly)
                    y2 = max(p[1] for p in poly)
                    area = max(1, (x2-x1)*(y2-y1))
                    boxes_scores.append(((x1,y1,x2,y2), float(area), None))

                merged = nms(boxes_scores, iou_thresh=IOU_MERGE_THRESHOLD)
                final = []
                for box, score, meta in merged:
                    x1,y1,x2,y2 = box
                    final.append([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

                detections[tpl] = final
                self._log(f"Final detections for {tpl}: {len(final)}")

                # optionally delete found template files
                if self.delete_found and final:
                    try:
                        os.remove(tpl_path)
                        self._log(f"Deleted template because found: {tpl}")
                    except Exception:
                        pass

            any_found = any(len(v)>0 for v in detections.values())
            if any_found or not self.result_only_when_found:
                self._log("Drawing & saving result image")
                self._draw_and_save_result(screen, detections)
            else:
                self._log("No detections — skipping result image creation")

            # coords file: clear/write readable summary
            lines = [f"[{now_ts()}] Coords summary", ""]
            for tpl, boxes in detections.items():
                lines.append(f"{tpl}:")
                if not boxes:
                    lines.append("  NOT FOUND")
                else:
                    for i,b in enumerate(boxes,1):
                        x1,y1 = b[0]
                        x2,y2 = b[2]
                        lines.append(f"  {i}. TopLeft=({x1},{y1})  BottomRight=({x2},{y2})")
                lines.append("")
            safe_write_overwrite(self.coords_path, "\n".join(lines)+"\n")

            self._finalize_logs()
            return detections

        except Exception:
            safe_append(self.log_path, f"[{now_ts()}] FATAL: {traceback.format_exc()}\n")
            return {}

# ----------------------------
# Command-line runner
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description='ImageFinder portable')
    p.add_argument('--no-delete', action='store_true', help='Do not delete templates when found')
    p.add_argument('--keep-ahk', action='store_true', help='Keep AhkStarter.log in logs folder')
    p.add_argument('--always-result', action='store_true', help='Always produce result_image.png even if no detections')
    p.add_argument('--no-draw', action='store_true', help='Do not draw result image (headless)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    finder = ImageFinder(delete_found=not args.no_delete, keep_ahk_log=args.keep_ahk, result_only_when_found=not args.always_result)
    res = finder.run()
    # minimal summary to stdout so user knows something happened
    found_total = sum(len(v) for v in res.values()) if res else 0
    print(f"ImageFinder run complete — templates={len(res)} matches_total={found_total}")
    