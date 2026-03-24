import cv2
import os
import csv
import time
import shutil
import tempfile
import glob
import threading
import pyttsx3
import queue as _queue
import numpy as np
from datetime import datetime
from deepface import DeepFace

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH = "employee_db"
ATTENDANCE_CSV = "attendance.csv"
LOG_CSV = "recognition_log.csv"
MODEL_NAME = "Facenet"
DETECTOR = "opencv"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.40
COOLDOWN_SEC = 30
RECOGNITION_FPS = 2
LATE_AFTER = "09:00"   # HH:MM — arrivals strictly after this are marked late
INTRUDER_DIR = "intruder_captures"
INTRUDER_CSV = "intruder_log.csv"

# ─── Anti-Spoofing Configuration ──────────────────────────────────────────────
ANTI_SPOOFING_ENABLED = True # Master switch
SPOOF_USE_DEEPFACE = True # Use DeepFace MiniFASNet
SPOOF_REAL_MIN_CONF = 0.55 # Below this confidence, treat verdict as uncertain
SPOOF_FAKE_MIN_CON = 0.40
SPOOF_LBP_THRESHOLD = 40.0
SPOOF_FREQ_THRESHOLD = 8.0
SPOOF_LOG_CSV = "spoof_log.csv"

# ──────────Face Detector Config──────────────────────────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if _face_cascade.empty():
    raise RuntimeError("Failed to load frontal face cascade - check OpenCv intallation")

_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml")
if _profile_cascade.empty():
    raise RuntimeError("Failes to load profile face cascade - check OpenCv intallation")

#────────────────────────────────────────────────────────────────────────────────
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(INTRUDER_DIR, exist_ok=True)
last_seen: dict = {}
_csv_lock = threading.Lock()

# ── Anti-Spoofing Engine ──────────────────────────────────────────────────────

def _lbp_texture_score(gray_face: np.ndarray) -> float:
    """
    Real faces have rich texture; printed photos / screens are flatter
    Higher score = more texture = more likely real
    """
    img = cv2.resize(gray_face, (64, 64)).astype(np.int16)

    # 8 neighbour offsets for LBP
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    center  = img[1:-1, 1:-1]
    lbp     = np.zeros_like(center, dtype=np.uint8)

    for bit, (dy, dx) in enumerate(offsets):
        ny = 1 + dy
        nx = 1 + dx
        neighbor = img[ny:ny + center.shape[0], nx:nx + center.shape[1]]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-6)

    mean = np.dot(hist, np.arange(256))
    variance = float(np.dot(hist, (np.arange(256) - mean) ** 2))

    # Scale to 0–100 for readability
    return min(variance / 50.0, 100.0)


def _fft_frequency_score(gray_face: np.ndarray) -> float:
    """
    Screen/printed photos lose high-frequency detail
    Higher score = more high-freq detail = more likely real
    """

    resized = cv2.resize(gray_face, (64, 64)).astype(np.float32)

    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)    
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    y_idx, x_idx = np.ogrid[:h, :w]

    low_mask = (y_idx - cy)**2 + (x_idx - cx)**2 <= radius**2
    low_energy = float(magnitude[low_mask].sum())
    total_energy = float(magnitude.sum()) + 1e-6
    high_ratio = (total_energy - low_energy) / total_energy * 100.0

    return high_ratio


def _check_face_geometry(color_crop: np.ndarray) -> tuple[bool, str]:
    """
    Detect extreme-angle presentations by checking two geometric properties
    that a real face satisfies but an angled phone/photo does not:

    1. Aspect ratio — check the ratio of height and width, low for tilted phones/photos
    2. Bilateral symmetry — a real face is roughly symmetric about its
       centre axis, tilted phone/photo looses this symmetry
    """

    if color_crop is None or color_crop.size == 0:
        return False, "empty crop"

    h, w = color_crop.shape[:2]

    # ── 1. Aspect-ratio──────────────────────────────────────────────────
    # real faces are roughly square, angled phone/photo can push aspect-ratio below 0.5 or above 2.0

    ratio = w / (h + 1e-6)
    if not (0.5 <= ratio <= 2.0):
        return False, f"suspicious aspect ratio {ratio:.2f} (expected 0.5–2.0)"

    # ── 2. Bilateral symmetry ─────────────────────────────────────────────────
    # Resize to a fixed square, split down the middle, mirror the right half,
    # and compare pixel-wise to the left half using normalised cross-correlation
    # real face scores ≥ 0.70, angled face scores lower

    SIZE = 64
    gray = cv2.cvtColor(cv2.resize(color_crop, (SIZE, SIZE)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mid = SIZE // 2
    left  = gray[:, :mid]
    right = np.fliplr(gray[:, mid:])  # mirror right half

    # computing and normalizing correlation
    l_norm = left - left.mean()
    r_norm = right - right.mean()
    denom = (np.linalg.norm(l_norm) * np.linalg.norm(r_norm)) + 1e-6
    symmetry_score = float(np.sum(l_norm * r_norm) / denom)

    SYMMETRY_THRESHOLD = 0.55
    if symmetry_score < SYMMETRY_THRESHOLD:
        return False, (
            f"low bilateral symmetry {symmetry_score:.2f} < {SYMMETRY_THRESHOLD} "
            f"— likely extreme-angle or profile presentation"
        )

    return True, f"geometry OK (ratio={ratio:.2f}, symmetry={symmetry_score:.2f})"


def _deepface_antispoof(face_crop: np.ndarray) -> tuple[bool, float]:
    """
    Run DeepFace anti-spoofing on cropped captured frame, cropping first prevents DeepFace from detecting a face in background
    """

    try:
        faces = DeepFace.extract_faces(
            img_path=face_crop,
            anti_spoofing=True,
            enforce_detection=False,
            detector_backend="skip",
        )

        if faces:
            face = faces[0]
            is_real = face.get("is_real", None)
            confidence = float(face.get("antispoof_score", 0.0))

            if is_real is not None:
                print(f"  [AntiSpoof-DeepFace] is_real={is_real}  score={confidence:.4f}")
                return bool(is_real), confidence
    
    except Exception as e:
        print(f"  [AntiSpoof-DeepFace error] {e}")

    return None, 0.0


def check_liveness_passive(color_crop: np.ndarray, gray_crop: np.ndarray) -> dict:
    """
    Passive liveness check on a pre-cropped frame
    Accepts the crop directly to avoid double face-detection disagreement

    Layers:
      1. DeepFace MiniFASNet on cropped frame
      2. LBP texture variance
      3. FFT high-frequency energy
    """

    result = {
        "live": False,
        "method": "unknown",
        "lbp": 0.0,
        "fft": 0.0,
        "deepface_real": None,
        "deepface_conf": 0.0,
        "reason": "",
    }

    if color_crop is None or color_crop.size == 0:
        result["reason"] = "Empty face crop"
        return result

    # ── Layer 0: Geometry gate (runs before any neural inference) ─────────
    # Rejects extreme-angle phone attacks that fool frontal-only models.
    
    geo_ok, geo_reason = _check_face_geometry(color_crop)

    print(f"  [AntiSpoof] Geometry: {geo_reason}")

    if not geo_ok:
        result["live"] = False
        result["method"] = "geometry"
        result["reason"] = f"Spoof (geometry): {geo_reason}"
        return result

    # Texture scores computed for logging/display
    lbp_score = _lbp_texture_score(gray_crop)
    fft_score = _fft_frequency_score(gray_crop)

    result["lbp"] = lbp_score
    result["fft"] = fft_score

    lbp_pass = lbp_score >= SPOOF_LBP_THRESHOLD
    fft_pass = fft_score  >= SPOOF_FREQ_THRESHOLD

    print(f"  [AntiSpoof] LBP={lbp_score:.1f}({'✓' if lbp_pass else '✗'})  "
          f"FFT={fft_score:.1f}({'✓' if fft_pass else '✗'})")

    # DeepFace on the pre-cropped frame
    df_real = None
    df_conf = 0.0

    if SPOOF_USE_DEEPFACE:
        df_real, df_conf = _deepface_antispoof(color_crop)
        result["deepface_real"] = df_real
        result["deepface_conf"] = df_conf

    # ── Decision ─────────────────────────────────────────────────────────────
    if df_real is not None:
        result["method"] = "deepface"

        if df_real:
            # Model outputs REAL — block if confidence is high enough to trust a SPOOF reversal
            result["live"] = True
            result["reason"] = (
                f"REAL — DeepFace(conf={df_conf:.2f})  "
                f"LBP={lbp_score:.1f}  FFT={fft_score:.1f}"
            )
        
        else:
            # Model outputs SPOOF
            confident_spoof = df_conf >= SPOOF_FAKE_MIN_CON

            if confident_spoof:
                result["live"] = False
                result["reason"] = f"Spoof: DeepFace=SPOOF (conf={df_conf:.2f})"

            else:
                # Uncertain — fall back to texture + frequency
                result["live"] = lbp_pass and fft_pass
                result["method"] = "deepface+texture"
                result["reason"] = (
                    f"Uncertain spoof (conf={df_conf:.2f}) — "
                    f"LBP={lbp_score:.1f}({'✓' if lbp_pass else '✗'})  "
                    f"FFT={fft_score:.1f}({'✓' if fft_pass else '✗'})"
                )

        print(f"  [AntiSpoof-Decision] live={result['live']}  {result['reason']}")
        print(f"  [DEBUG] LBP={lbp_score:.1f}  FFT={fft_score:.1f}")
    
    else:
        
        all_pass = lbp_pass and fft_pass
        result["live"] = all_pass
        result["method"] = "texture+freq"

        if all_pass:
            result["reason"] = f"REAL — LBP={lbp_score:.1f} ✓  FFT={fft_score:.1f} ✓"

        else:
            fails = []

            if not lbp_pass:
                fails.append(f"LBP={lbp_score:.1f}<{SPOOF_LBP_THRESHOLD}")
            
            if not fft_pass:
                fails.append(f"FFT={fft_score:.1f}<{SPOOF_FREQ_THRESHOLD}")
            
            result["reason"] = "Spoof: " + "  ".join(fails)

    return result


def log_spoof_attempt(frame, liveness: dict):
    """Save snapshot and write to spoof_log.csv."""

    dt = datetime.now()
    img_path = ""

    fname = dt.strftime("spoof_%Y%m%d_%H%M%S.jpg")
    img_path = os.path.join(INTRUDER_DIR, fname)

    try:
        cv2.imwrite(img_path, frame)
    
    except Exception:
        pass

    
    with _csv_lock:
        if not os.path.exists(SPOOF_LOG_CSV):
            with open(SPOOF_LOG_CSV, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "Date","Time","Method","LBP","FFT",
                    "DeepFace_Real","DeepFace_Conf","Reason","Image"
                ])

        with open(SPOOF_LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"),
                liveness["method"],
                f"{liveness['lbp']:.2f}",
                f"{liveness['fft']:.2f}",
                str(liveness["deepface_real"]),
                f"{liveness['deepface_conf']:.3f}",
                _sanitise(liveness["reason"]),
                img_path,
            ])
    
    return img_path


# ── Text-to-speech ───────────────────────────────────────────────────────────


_tts_queue: _queue.Queue = _queue.Queue()

def _tts_worker():
    """Runs once for the lifetime of a process."""

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    
    except Exception:
        return
    
    while True:
        text = _tts_queue.get()

        if text is None:
            break

        try:
            engine.say(text)
            engine.runAndWait()
        
        except Exception:
            pass
        _tts_queue.task_done()

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def _speak(text: str):
    """Queue a TTS utterance"""
    _tts_queue.put(text)

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _sanitise(text: str) -> str:
    """Prevents CSV file corruption by encoding it to UTF-8"""

    text = text.encode("utf-8", errors="replace").decode("utf-8")
    if text and text[0] in ('=', '+', '-', '@', '\t', '\r'):
        text = "'" + text
    return text


def ensure_csvs():
    '''if required csv files exist load them else create them'''

    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Employee ID", "Name", "Date", "Time", "Status"])

    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Date", "Time", "Employee ID", "Name", "Result", "Distance"])

    if not os.path.exists(SPOOF_LOG_CSV):
        with open(SPOOF_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "Date","Time","Method","LBP","FFT",
                "DeepFace_Real","DeepFace_Conf","Reason","Image"
            ])

def log_event(emp_id: str, name: str, result: str, distance: float):
    """log every detection event to recognition_log.csv."""

    dt = datetime.now()
    
    with _csv_lock:
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"),
                emp_id, _sanitise(name), result,
                f"{distance:.4f}"
            ])

def mark_attendance(emp_id: str, name: str, distance: float) -> bool:
    """Log every attempt; write attendance only outside cooldown."""

    log_event(emp_id, name, "MATCH", distance)

    now = time.time()

    if emp_id in last_seen and (now - last_seen[emp_id]) < COOLDOWN_SEC:
        return False          # already marked recently
    
    last_seen[emp_id] = now
    dt = datetime.now()
    
    with _csv_lock:
        with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [emp_id, _sanitise(name),
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"), "Present"
                ])
    
    print(f"  Marked: {name} ({emp_id}) at {dt.strftime('%H:%M:%S')}")
    return True

def mark_unknown(distance: float, frame=None) -> str:
    """Save snapshot of unauthorized person and log to intruder_log.csv."""

    dt = datetime.now()
    img_path = ""

    if frame is not None:
        fname = dt.strftime("intruder_%Y%m%d_%H%M%S.jpg")
        img_path = os.path.join(INTRUDER_DIR, fname)
        cv2.imwrite(img_path, frame)

    
    with _csv_lock:
        if not os.path.exists(INTRUDER_CSV):
            with open(INTRUDER_CSV, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Date", "Time", "Image Path", "Distance"])

        with open(INTRUDER_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"),
                img_path,
                f"{distance:.4f}",
            ])

    log_event("UNKNOWN", "Unknown", "DENIED", distance)
    return img_path

# ── Misc helpers ──────────────────────────────────────────────────────────────

def clear_deepface_cache():
    """Remove stale DeepFace .pkl cache files from DB_PATH."""
    for pkl in glob.glob(os.path.join(DB_PATH, "*.pkl")):
        os.remove(pkl)
        print(f"  Removed stale cache: {pkl}")

def parse_employee_from_path(img_path: str):
    '''retrieve name and id of employee from the stored image path'''

    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")

    emp_id = parts[0]

    if len(parts) > 2 and parts[-1].isdigit():
        name = "_".join(parts[1:-1])

    elif len(parts) > 1:
        name = "_".join(parts[1:])
        
    else:
        name = stem

    return emp_id, name

def get_enrolled_employees():
    '''retrieve unique employees from the stored image paths'''

    seen = {}
    if not os.path.isdir(DB_PATH):
        return []

    for fname in os.listdir(DB_PATH):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            emp_id, name = parse_employee_from_path(fname)
            if emp_id not in seen:
                seen[emp_id] = {"id": emp_id, "name": name, "file": fname}

    return list(seen.values())

# ── Status bar ────────────────────────────────────────────────────────

def draw_status_bar(frame, status: dict):
    """
    Draws a full-width status bar at the bottom of the scanner window, fades after 4 secs
    Colours: green=matched, red=denied, orange=spoof detected
    """

    if not status or not status.get("text"):
        return

    age = time.time() - status["ts"]
    if age > 4.0:
        return

    h, w = frame.shape[:2]
    matched = status.get("matched", False)
    is_spoof = status.get("spoof", False)

    # bar fade out
    alpha = 1.0 if age < 3.0 else max(0.0, 1.0 - (age - 3.0))

    bar_h = 72
    overlay = frame.copy()

    # Bar background — orange for spoof, green for match, red for unknown
    if is_spoof:
        bar_color = (0, 100, 200)
        accent    = (0, 165, 255)

    elif matched:
        bar_color = (30, 140, 50)
        accent    = (60, 210, 90)
    
    else:
        bar_color = (30, 30, 180)
        accent    = (80, 80, 230)

    #overlay bar on frame
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), bar_color, -1)
    # Accent strip at top of bar
    cv2.rectangle(overlay, (0, h - bar_h), (w, h - bar_h + 3), accent, -1)

    # Employee photo thumbnail if found
    #if not found: display '+' if employee enrolled else '-'

    thumb_size = bar_h - 10
    thumb_x = 5
    thumb_y = h - bar_h + 5
    img_path = status.get("img_path", "")
    drawn_thumb = False

    if img_path and os.path.isfile(img_path):

        try:
            photo = cv2.imread(img_path)

            if photo is not None:
                photo = cv2.resize(photo, (thumb_size, thumb_size))
                mask = np.zeros((thumb_size, thumb_size), dtype=np.uint8)

                cx = cy = thumb_size // 2
                cv2.circle(mask, (cx, cy), cx, 255, -1)

                for c in range(3):
                    overlay[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size, c] = np.where(
                        mask == 255,
                        photo[:, :, c],
                        overlay[thumb_y:thumb_y+thumb_size, thumb_x:thumb_x+thumb_size, c]
                    )

                cv2.circle(overlay, (thumb_x + cx, thumb_y + cy), cx, accent, 2)
                drawn_thumb = True
        
        except Exception as e:
            print(e)

    if not drawn_thumb:
        icon_cx = thumb_x + thumb_size // 2
        icon_cy = thumb_y + thumb_size // 2

        cv2.circle(overlay, (icon_cx, icon_cy), thumb_size // 2, accent, -1)
        icon_char = "+" if matched else ("!" if is_spoof else "x")
        (iw, ih), _ = cv2.getTextSize(icon_char, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        cv2.putText(overlay, icon_char,
                    (icon_cx - iw // 2, icon_cy + ih // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    text_x = thumb_x + thumb_size + 10

    # Main text
    main_text = status["text"]
    cv2.putText(overlay, main_text,
                (text_x, h - bar_h + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    # Sub text
    sub_text = status.get("sub", "")
    if sub_text:
        cv2.putText(overlay, sub_text,
                    (text_x, h - bar_h + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_hud(frame, enrolled_count: int):
    """Top-left corner info strip"""
    
    _, w = frame.shape[:2]

    dt = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
    cv2.putText(frame, f"FaceGuard   |   {dt}   |   Enrolled: {enrolled_count}   |   Q = quit",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

# ── Background recognition worker ─────────────────────────────────────────────

'''manages a dedicated background thread that runs independently of main loop'''

class RecognitionWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._input_frame = None
        self._result = None
        self._busy = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            if not self._busy:
                self._input_frame = frame.copy()
                self._busy = True

    def get_result(self):
        with self._lock:
            return self._result

    def _loop(self):
        while True:
            frame = None

            with self._lock:
                if self._busy and self._input_frame is not None:
                    frame = self._input_frame
                    self._input_frame = None
            
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                result = self._recognize(frame)
            except Exception as e:
                print(f"  [RecognitionWorker error] {e}")
                result = None
            with self._lock:
                self._result = result
                self._busy   = False

    def _recognize(self, frame):
        """
        Detect face → run anti-spoof → run face recognition"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Profile cascade check: if a side-face is visible but not a frontal face → Block immediately.
        if len(faces) == 0:
            profiles = _profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )

            if len(profiles) > 0:
                now = datetime.now()
                ts  = time.time()

                reason = "profile/extreme-angle face detected — likely angled phone attack"
                liveness = {
                    "live": False,
                    "method": "geometry",
                    "reason": reason,
                    "lbp": 0.0,
                    "fft": 0.0,
                    "deepface_real": None,
                    "deepface_conf": 0.0
                }

                spoof_img = log_spoof_attempt(frame, liveness)
                log_event("SPOOF", "Spoof Attempt", "SPOOF", 0.0)
                _speak("Spoof detected")
                print(f"  [AntiSpoof] BLOCKED — {reason}")

                return {
                    "text": "SPOOF DETECTED:  Presentation Attack",
                    "sub":  f"{reason}  |  {now.strftime('%H:%M:%S')}",
                    "matched": False, "spoof": True, "ts": ts,
                    "img_path": spoof_img, "liveness": liveness,
                }
            
            return None

        # Crop the largest face once
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add 20% padding so DeepFace has enough context around the face
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)

        fh, fw = frame.shape[:2]

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        color_crop = frame[y1:y2, x1:x2]
        gray_crop  = gray[y1:y2, x1:x2]

        now = datetime.now()
        ts  = time.time()

        try:
            # ── Step 1: Anti-Spoofing Gate ────────────────────────────────────
            if ANTI_SPOOFING_ENABLED:

                try:
                    liveness = check_liveness_passive(color_crop, gray_crop)

                except Exception as e:
                    print(f"  [AntiSpoof error] {e}")

                    liveness = {
                        "live": False,
                        "method": "error",
                        "reason": "check failed - access denied",
                        "lbp": 0.0,
                        "fft": 0.0,
                        "deepface_real": None,"deepface_conf": 0.0
                    }

                if not liveness["live"]:
                    spoof_img = log_spoof_attempt(frame, liveness)
                    log_event("SPOOF", "Spoof Attempt", "SPOOF", 0.0)
                    _speak("Spoof detected")
                    print(f"  [AntiSpoof] BLOCKED — {liveness['reason']}")

                    return {
                        "text": "SPOOF DETECTED:  Presentation Attack",
                        "sub":  f"{liveness['reason']}  |  {now.strftime('%H:%M:%S')}",
                        "matched": False,
                        "spoof": True,
                        "ts": ts,
                        "img_path": spoof_img,
                        "liveness": liveness,
                    }

            # ── Step 2: Face Recognition (if liveness passed) ───
            matched = False
            emp_id = ""
            name = ""
            distance = 1.0
            top = None

            results = DeepFace.find(
                img_path = frame,
                db_path = DB_PATH,
                model_name = MODEL_NAME,
                detector_backend = DETECTOR,
                distance_metric = DISTANCE_METRIC,
                enforce_detection = False,
                silent = True,
            )

            if results and len(results[0]) > 0:
                top = results[0].iloc[0]
                dist_col = next((c for c in top.index if DISTANCE_METRIC.lower() in c.lower()), None)

                if dist_col is None:
                    numeric_cols = [
                        c for c in top.index
                        if c != "identity" and isinstance(top[c], (int, float))
                    ]

                    dist_col = numeric_cols[0] if numeric_cols else None
                
                distance = float(top[dist_col]) if dist_col else 1.0

                if distance <= THRESHOLD:
                    emp_id, name = parse_employee_from_path(str(top["identity"]))
                    matched = True

        except Exception as e:
            print(f"  [DeepFace error] {e}")
        
        # Build liveness info for status bar
        liveness_sub = ""

        if ANTI_SPOOFING_ENABLED:
            lv = locals().get("liveness", {})

            if lv:
                if lv.get("method") == "deepface":
                    liveness_sub = f"Liveness ✓ DeepFace (conf={lv['deepface_conf']:.2f})"

                else:
                    liveness_sub = (
                        f"Liveness ✓  LBP={lv.get('lbp',0):.0f}  "
                        f"FFT={lv.get('fft',0):.0f}"
                    )

        if matched:
            newly = mark_attendance(emp_id, name, distance)
            status_text = f"ACCESS GRANTED:  {name}  ({emp_id})"
            sub = (f"Attendance recorded at {now.strftime('%H:%M:%S')}"
                   if newly else
                   f"Already marked today:  {now.strftime('%H:%M:%S')}")
            
            if liveness_sub:
                sub += f"  |  {liveness_sub}"
            
            _speak("Access Granted")

            return {
                "text": status_text, "sub": sub,
                "matched": True, "spoof": False, "ts": ts,
                "img_path": str(top["identity"]),
            }

        else:
            intruder_img = mark_unknown(distance, frame)

            _speak("Access Denied")

            sub = f"No match found:  {now.strftime('%H:%M:%S')}"

            if liveness_sub:
                sub += f"  |  {liveness_sub}"
            
            return {
                "text": "ACCESS DENIED:  Unregistered Person",
                "sub": sub,
                "matched": False, "spoof": False, "ts": ts,
                "img_path": intruder_img,
            }

# ── Late report ───────────────────────────────────────────────────────────────

def generate_late_report() -> str | None:
    """
    Scan attendance.csv for today's entries that arrived after designated time and log it.
    """

    if not os.path.exists(ATTENDANCE_CSV):
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    late_rows = []

    with open(ATTENDANCE_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("Date") != today:
                continue

            try:
                fmt = "%H:%M:%S" if len(row.get("Time", "")) > 5 else "%H:%M"
                arrival = datetime.strptime(row["Time"], fmt).time()
                c_fmt = "%H:%M:%S" if len(LATE_AFTER) > 5 else "%H:%M"
                cutoff = datetime.strptime(LATE_AFTER, c_fmt).time()
            
            except ValueError:
                continue
            
            if arrival > cutoff:
                late_rows.append({
                    "Employee ID": row["Employee ID"],
                    "Name": row["Name"],
                    "Date": row["Date"],
                    "Arrival Time": row["Time"],
                    "Late After": LATE_AFTER,
                })

    if not late_rows:
        return None

    out_path = f"late_{today}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Employee ID","Name","Date","Arrival Time","Late After"])
        writer.writeheader()
        writer.writerows(late_rows)

    return out_path

# ── Attendance camera ─────────────────────────────────────────────────────────

def run_attendance():
    employees = get_enrolled_employees()
    if not employees:
        print("  No employees enrolled. Please enroll first.")
        return

    clear_deepface_cache()
    print(f"\n  Starting system ({len(employees)} employee(s) enrolled)")
    print("  Press  Q  to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Cannot open webcam (try changing VideoCapture(0) to (1))")
        return

    worker = RecognitionWorker()
    current_status = None
    last_submit = 0.0
    submit_interval = 1.0 / RECOGNITION_FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  Lost camera feed.")
            break

        frame = cv2.flip(frame, 1)

        # Submit to background at controlled rate
        now = time.time()
        if now - last_submit >= submit_interval:
            worker.submit(frame)
            last_submit = now

        # Pull latest recognition result
        latest = worker.get_result()
        if latest is not None:
            current_status = latest

        draw_hud(frame, len(employees))
        draw_status_bar(frame, current_status)

        cv2.imshow("Facial Attendance System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Camera closed.")


# ── Enrollment ────────────────────────────────────────────────────────────────
'''enrolling a new employee thorugh an saved image or through capture at checkpoint'''

ENROLL_DUP_THRESHOLD = THRESHOLD   # 1:N duplicate threshold

def _is_face_duplicate_1n(candidate_path: str) -> tuple:
    """Forced 1:N search to capture same person across different captures"""

    if not os.path.isdir(DB_PATH) or not os.listdir(DB_PATH):
        return False, "", "", 1.0

    # CACHE CLEAR before checking
    clear_deepface_cache() 

    try:
        results = DeepFace.find(
            img_path=candidate_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            silent=True
        )

        if results and len(results[0]) > 0:
            top = results[0].iloc[0]
            dist_col = next((c for c in top.index if DISTANCE_METRIC.lower() in c.lower()), None)
            distance = float(top[dist_col]) if dist_col else 1.0

            if distance <= ENROLL_DUP_THRESHOLD:
                matched_id, matched_name = parse_employee_from_path(str(top["identity"]))
                return True, matched_id, matched_name, distance

    except Exception as e:
        print(f"  [1:N Check Error] {e}")
    
    return False, "", "", 1.0


def enroll_employee():

    print("\n── Enroll New Employee ──────────────────────────────────")
    emp_id = input("  Enter Employee ID: ").strip()
    name = input("  Enter Employee Name: ").strip().replace(" ", "_")

    if not emp_id or not name:
        print("  ID and Name cannot be empty.")
        return

    for emp in get_enrolled_employees():
        if emp["id"].lower() == emp_id.lower():
            print(f"  Employee ID '{emp_id}' already enrolled. Aborting.")
            return
        if emp["name"].lower() == name.lower():
            print(f"  Name '{name}' already enrolled. Aborting.")
            return

    # Use a temp path for the first photo — duplicate check only
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    captured_paths = _enroll_from_webcam(tmp_path, name, emp_id)

    if not captured_paths:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print("  No photos captured. Enrollment cancelled.")
        return

    # Duplicate check on first photo only
    print("  Running duplicate check on first photo...")
    is_dup, dup_id, dup_name, dist = _is_face_duplicate_1n(captured_paths[0])

    if is_dup:
        for p in captured_paths:
            if os.path.exists(p):
                os.remove(p)
        print(f"  Duplicate detected — already enrolled as '{dup_name}' ({dup_id}). Aborting.")
        return

    # Move all captured photos to DB_PATH
    for i, src in enumerate(captured_paths):
        dest = os.path.join(DB_PATH, f"{emp_id}_{name}_{i + 1}.jpg")
        shutil.move(src, dest)
        print(f"  Saved -> {dest}")

    clear_deepface_cache()
    print(f"  Enrolled: {name.replace('_', ' ')} ({emp_id}) — {len(captured_paths)} photos")


def _enroll_from_webcam(dest_path: str, name: str, emp_id: str, num_photos: int = 5):
    '''Capture multiple photos for enrollment'''

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Cannot open webcam.")
        return []

    print(f"  Capturing {num_photos} photos. Press SPACE to capture | ESC to cancel")

    captured_paths = []
    photo_num = 0

    while photo_num < num_photos:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 90), 2)

        color = (0, 220, 90) if len(faces) == 1 else (0, 140, 255)

        cv2.putText(
            display,
            f"Enrolling: {name.replace('_', ' ')}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Photo {photo_num + 1}/{num_photos} : try a different angle",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            display,
            "SPACE=capture  ESC=cancel",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1
        )

        cv2.imshow("Enroll Employee", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("  Enrollment cancelled.")
            break

        if key == 32:
            if len(faces) == 0:
                print("  No face detected — move closer or improve lighting.")
                continue
            if len(faces) > 1:
                print("  Multiple faces — ensure only one person is visible.")
                continue

            # Build path: empid_empname_N.jpg
            base = os.path.splitext(dest_path)[0]
            photo_path = f"{base}_{photo_num + 1}.jpg"
            cv2.imwrite(photo_path, frame)
            captured_paths.append(photo_path)
            print(f"  Photo {photo_num + 1}/{num_photos} saved -> {photo_path}")
            photo_num += 1

            if photo_num < num_photos:
                
                for i in range(60):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame,
                                f"Good! Adjust pose for photo {photo_num + 1}/{num_photos}",
                                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 90), 2)
                    cv2.imshow("Enroll Employee", frame)
                    cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    return captured_paths


# ── Reports ───────────────────────────────────────────────────────────────────

def view_attendance():
    '''function to view today's attendance logs'''

    if not os.path.exists(ATTENDANCE_CSV):
        print("  No attendance records yet.")
        return

    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n  Attendance - {today}")
    print(f"  {'ID':<10} {'Name':<22} {'Time':<10} Status")
    print("  " + "-" * 50)

    count = 0
    
    with open(ATTENDANCE_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("Date") == today:
                print(f"  {row['Employee ID']:<10} {row['Name']:<22} "
                      f"{row['Time']:<10} {row['Status']}")
                count += 1
    
    print(f"\n  Total: {count} record(s) today.\n")


def view_log():
    '''view today's stored logs'''

    if not os.path.exists(LOG_CSV):
        print("  No log entries yet.")
        return
    
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n  Recognition Log - {today}")
    print(f"  {'Time':<10} {'ID':<10} {'Name':<20} {'Result':<8} Distance")
    print("  " + "-" * 58)

    count = 0

    with open(LOG_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("Date") == today:
                print(f"  {row['Time']:<10} {row['Employee ID']:<10} "
                      f"{row['Name']:<20} {row['Result']:<8} {row['Distance']}")
                count += 1
    
    print(f"\n  Total: {count} event(s) today.\n")


def list_employees():
    '''print list of all employees'''

    employees = get_enrolled_employees()

    print(f"\n  Enrolled Employees ({len(employees)})")
    print(f"  {'ID':<10} {'Name':<25} File")
    print("  " + "-" * 55)

    for emp in employees:
        print(f"  {emp['id']:<10} {emp['name'].replace('_',' '):<25} {emp['file']}")
    print()


# ── Intruder report ──────────────────────────────────────────────────────────

def generate_intruder_report() -> str | None:
    """Export today's intruder log entries to a dated CSV file."""

    if not os.path.exists(INTRUDER_CSV):
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []

    with open(INTRUDER_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("Date") == today:
                rows.append(row)

    if not rows:
        return None

    out_path = f"intruder_report_{today}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Time", "Image Path", "Distance"])
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ensure_csvs()
    print("\n" + "=" * 55)
    print("   Facial Recognition Attendance System")
    print("=" * 55)

    while True:
        print("\n  [1]  Enroll new employee")
        print("  [2]  Start system")
        print("  [3]  View today's logs")
        print("  [4]  View today's recognition log")
        print("  [5]  List enrolled employees")
        print("  [6]  Generate late arrivals report")
        print("  [7]  Generate intruder report")
        print("  [8]  Exit")
        choice = input("\n  Select option: ").strip()

        if   choice == "1": enroll_employee()
        elif choice == "2": run_attendance()
        elif choice == "3": view_attendance()
        elif choice == "4": view_log()
        elif choice == "5": list_employees()
        elif choice == "6":
            path = generate_late_report()
            if path:
                print(f"  Late report saved -> {path}  (cutoff: {LATE_AFTER})")
            else:
                print(f"  No late arrivals today (cutoff: {LATE_AFTER})")
        
        elif choice == "7":
            path = generate_intruder_report()
            if path:
                print(f"  Intruder report saved -> {path}")
            else:
                print(f"  No intruder attempts recorded today.")
        
        elif choice == "8": print("\n  Goodbye!\n"); break
        else: print("  Invalid option.")


if __name__ == "__main__":
    main()
