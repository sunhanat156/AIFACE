import cv2
import numpy as np
import threading
import time
import datetime
import os
import json
import onnxruntime as ort
from insightface import model_zoo
from insightface.app import FaceAnalysis
from backend.database.models import User, AccessLog, get_db
from backend.core import config_store
from backend.paths import APP_DIR, INSIGHTFACE_ROOT

# Directory to store captured face snapshots
CAPTURED_DIR = os.path.join(APP_DIR, "captured_faces")
os.makedirs(CAPTURED_DIR, exist_ok=True)

EMBEDDING_DIM = 512

# A button-triggered entry scan only trusts a detection that's both confident
# and close enough to the camera — a small/blurry/off-angle face produces a
# noisier embedding that can drift close to an unrelated person's vector.
MIN_DET_SCORE = 0.65
MIN_FACE_SIZE = 80  # px, shorter side of the detection box

# ── Shared face engine (GPU via DirectML, falls back to CPU automatically) ─
# insightface's SCRFD detector finds the box + 5-point alignment landmarks in
# the same pass ArcFace uses for the embedding, so one FaceAnalysis instance
# covers both enrolment stills and the live video loop — unlike the old
# MediaPipe (detect) + dlib (encode) pair, everything here runs on one ONNX
# Runtime session per model, and DirectML puts that on the GPU.
_face_app = FaceAnalysis(
    name="buffalo_l",
    root=INSIGHTFACE_ROOT,
    allowed_modules=["detection", "recognition"],
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
)
_face_app.prepare(ctx_id=0, det_size=(640, 640))
_face_lock = threading.Lock()


def _get_faces(img_bgr):
    """Faces found by insightface in a BGR frame, largest first.

    insightface (and the DirectML session behind it) is not guaranteed
    thread-safe across concurrent calls, and both the live video loop and
    enrolment can call in at the same time.
    """
    with _face_lock:
        faces = _face_app.get(img_bgr)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces


# ── Idle gates for the auto-scan (keep the GPU asleep while nobody's there) ─
# 1. scene_changed: frame differencing on a tiny blurred thumbnail (~0.1 ms).
# 2. face_present: the same SCRFD detector, but a CPU-only session at 256px
#    (~13 ms, caught 15/15 real enrolment faces; a Haar cascade caught 2/15).
# Only once both pass does the GPU engine (detect 640 + ArcFace) run.
_presence_opts = ort.SessionOptions()
_presence_opts.intra_op_num_threads = 4
_presence_det = model_zoo.get_model(
    _face_app.det_model.model_file,
    providers=["CPUExecutionProvider"], sess_options=_presence_opts,
)
_presence_det.prepare(ctx_id=-1, det_thresh=0.5, input_size=(256, 256))
_presence_lock = threading.Lock()

MOTION_PIXEL_DIFF = 25      # grey-level change that counts as "moved"
MOTION_MIN_FRACTION = 0.002  # share of the 160x120 thumbnail that must move
_motion_ref = None
_motion_lock = threading.Lock()


def scene_changed(img_bgr):
    """True if the frame differs from the previous one checked (and on the
    first call). Always updates the reference frame."""
    global _motion_ref
    small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img_bgr, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    with _motion_lock:
        ref, _motion_ref = _motion_ref, small
    if ref is None:
        return True
    moved = np.count_nonzero(cv2.absdiff(small, ref) > MOTION_PIXEL_DIFF)
    return moved > small.size * MOTION_MIN_FRACTION


def face_present(img_bgr):
    """True if any face is in the frame — CPU only, no recognition."""
    if img_bgr is None or img_bgr.size == 0:
        return False
    with _presence_lock:
        boxes, _ = _presence_det.detect(img_bgr, max_num=1)
    return len(boxes) > 0


# Set by the auto-scan: with nobody in view the dashboard preview drops to
# IDLE_PREVIEW_FPS, since a full-rate stream of an empty room is wasted work.
IDLE_PREVIEW_FPS = 10
preview_idle = False


def _expand_box(x1, y1, x2, y2, img_w, img_h, padding=0.25):
    """Grow a tight detection box a bit, for nicer crops/snapshots."""
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * padding, bh * padding
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(img_w, int(x2 + px)),
        min(img_h, int(y2 + py)),
    )


def encode_faces(img_bgr, max_faces=1):
    """Return up to `max_faces` face embeddings (largest first) from a BGR image."""
    faces = _get_faces(img_bgr)
    return [f.normed_embedding for f in faces[:max_faces]]


def _augment_image(img_bgr):
    """
    Generate augmented versions of an image to create more encodings per photo.
    This significantly increases recognition accuracy at different angles.
    Returns list of augmented BGR numpy arrays.
    """
    augmented = [img_bgr]
    h, w = img_bgr.shape[:2]

    # 1. Horizontal flip (simulates mirror)
    augmented.append(cv2.flip(img_bgr, 1))

    # 2. Slight brightness variations
    for alpha in [0.8, 1.2]:
        bright = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0)
        augmented.append(bright)

    # 3. Small rotations (-15, +15 degrees)
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)

    # 4. Slight scale crop (zoom in 10%)
    crop_pct = 0.05
    cy, cx = int(h * crop_pct), int(w * crop_pct)
    cropped = img_bgr[cy:h - cy, cx:w - cx]
    cropped = cv2.resize(cropped, (w, h))
    augmented.append(cropped)

    return augmented


def extract_multi_encodings(img_bgr):
    """
    Extract multiple face encodings from a single image using augmentation.
    Returns list of 512-d numpy arrays.
    """
    all_encodings = []
    augmented_images = _augment_image(img_bgr)

    for aug_img in augmented_images:
        encs = encode_faces(aug_img)
        if encs:
            all_encodings.append(encs[0])

    return all_encodings


def extract_multi_encodings_from_files(image_bytes_list):
    """
    Process multiple uploaded images and extract all possible encodings.
    Each image is also augmented for maximum coverage.
    Returns (list_of_encodings, count_of_valid_images).
    """
    all_encodings = []
    valid_count = 0

    for img_bytes in image_bytes_list:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Check there's a face first
        if not encode_faces(img):
            continue

        valid_count += 1

        # Now extract augmented encodings
        encs = extract_multi_encodings(img)
        all_encodings.extend(encs)

    return all_encodings, valid_count


class FaceDetector:
    def __init__(self):
        self.known_face_encodings = []  # list of 512-d np arrays (L2-normalized)
        self.known_face_matrix = None   # stacked (N, 512) for fast batched cosine sim
        self.known_face_names = []
        self.departments = {}  # user name -> department, for tagging passive detections
        self.logged_names = {}
        self.recent_activity = []
        self.activity_lock = threading.Lock()
        cfg = config_store.load_config()
        self.tolerance = float(cfg["tolerance"])
        self.log_cooldown = int(cfg["log_cooldown"])
        self.unknown_cooldown = int(cfg["unknown_cooldown"])
        # load_known_faces() called at startup

    def apply_settings(self, cfg=None):
        """Re-read tuning values from config.json so the Settings page takes
        effect immediately, without restarting the server."""
        cfg = cfg or config_store.load_config()
        try:
            self.tolerance = float(cfg.get("tolerance", self.tolerance))
            self.log_cooldown = int(cfg.get("log_cooldown", self.log_cooldown))
            self.unknown_cooldown = int(cfg.get("unknown_cooldown", self.unknown_cooldown))
        except (TypeError, ValueError) as e:
            print(f"[WARN] Invalid setting value: {e}")
        print(f"[INFO] Settings applied: tolerance={self.tolerance}, cooldown={self.log_cooldown}s")

    def load_known_faces(self):
        """Load all registered face encodings from the database."""
        try:
            db = next(get_db())
            users = db.query(User).all()
            temp_encodings = []
            temp_names = []
            temp_departments = {}
            skipped = 0
            for user in users:
                temp_departments[user.name] = user.department or ""
                if user.face_encodings:
                    try:
                        encs = json.loads(user.face_encodings)
                        for enc in encs:
                            arr = np.array(enc, dtype=np.float32)
                            # Old encodings were 128-d (dlib); the new GPU engine
                            # produces 512-d vectors. A mixed batch can't be
                            # compared, so stale ones are skipped until the
                            # person is re-registered with the new engine.
                            if arr.shape[0] != EMBEDDING_DIM:
                                skipped += 1
                                continue
                            temp_encodings.append(arr)
                            temp_names.append(user.name)
                    except Exception as e:
                        print(f"Error loading encoding for {user.name}: {e}")
            self.known_face_encodings = temp_encodings
            self.known_face_matrix = np.stack(temp_encodings) if temp_encodings else None
            self.known_face_names = temp_names
            self.departments = temp_departments
            print(f"[INFO] Loaded {len(self.known_face_names)} face vectors for {len(set(self.known_face_names))} users.")
            if skipped:
                print(f"[WARN] Skipped {skipped} face vector(s) from an older engine — re-register those users.")
        except Exception as e:
            print(f"[ERROR] load_known_faces: {e}")

    def _save_face_snapshot(self, face_bgr, name):
        """Save a cropped face image to disk and return the filename."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
        filename = f"{safe_name}_{ts}.jpg"
        filepath = os.path.join(CAPTURED_DIR, filename)
        cv2.imwrite(filepath, face_bgr)
        return filename

    def _match_face(self, embedding):
        """
        Match a face embedding against the known database using cosine
        similarity (both sides are L2-normalized, so a dot product IS the
        cosine similarity). Returns (name, confidence%).
        """
        if self.known_face_matrix is None:
            return "Unknown", 0.0

        sims = self.known_face_matrix @ embedding
        distances = 1.0 - sims

        best_idx = int(np.argmin(distances))
        if distances[best_idx] >= self.tolerance:
            return "Unknown", 0.0

        # Group by name and keep the best (lowest distance) match per person
        name_scores = {}
        for i, d in enumerate(distances):
            if d < self.tolerance:
                person = self.known_face_names[i]
                if person not in name_scores or d < name_scores[person]:
                    name_scores[person] = d

        best_name = min(name_scores, key=name_scores.get)
        best_dist = name_scores[best_name]
        # np.float32 isn't JSON-serializable by FastAPI's encoder, so cast
        # back to a plain Python float before it goes anywhere near a response.
        confidence = float(max(0, min(100, (1 - best_dist / 0.6) * 100)))

        return best_name, confidence

    def verify_identity(self, claimed_name, img_bgr):
        """Self-service check-in: confirm the face in `img_bgr` actually
        belongs to `claimed_name` (the user picked on the Check-In page),
        rather than just recognising whoever is in frame.

        Returns (ok, matched_name, confidence, face_bgr_crop_or_None).
        """
        if img_bgr is None or img_bgr.size == 0:
            return False, "Unknown", 0.0, None

        faces = _get_faces(img_bgr)
        if not faces:
            return False, "Unknown", 0.0, None

        face = faces[0]
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = _expand_box(*face.bbox.astype(int), w, h, padding=0.25)

        matched_name, confidence = self._match_face(face.normed_embedding)
        face_crop = img_bgr[y1:y2, x1:x2]
        ok = matched_name == claimed_name and matched_name != "Unknown"
        return ok, matched_name, confidence, (face_crop.copy() if face_crop.size > 0 else None)

    def identify_once(self, img_bgr):
        """Button-triggered entry scan: identify whoever is in front of the
        camera right now, from a single fresh frame.

        A quality gate (detector confidence + minimum face size) comes before
        the identity match — a poor detection is always reported as Unknown
        rather than being handed to _match_face, since a noisy embedding from
        a tiny/blurry/off-angle face is what used to let a stranger through.
        """
        if img_bgr is None or img_bgr.size == 0:
            return "Unknown", 0.0, None, None

        faces = _get_faces(img_bgr)
        if not faces:
            return "Unknown", 0.0, None, None

        face = faces[0]
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        box = (x1, y1, x2, y2)
        if face.det_score < MIN_DET_SCORE or min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
            return "Unknown", 0.0, None, box

        h, w = img_bgr.shape[:2]
        px1, py1, px2, py2 = _expand_box(x1, y1, x2, y2, w, h, padding=0.25)
        name, confidence = self._match_face(face.normed_embedding)
        face_crop = img_bgr[py1:py2, px1:px2]
        return name, confidence, (face_crop.copy() if face_crop.size > 0 else None), box

    def log_access(self, name, status, confidence, face_bgr=None,
                    check_type="DETECT", department=None, latitude=None, longitude=None):
        """Log a face detection event to the database with face snapshot."""
        now = datetime.datetime.now()
        cooldown = self.log_cooldown if name != "Unknown" else self.unknown_cooldown
        # Clock in/out events come from an explicit user action on the
        # Check-In page, so they always get logged even inside the passive
        # dashboard's cooldown window.
        if check_type == "DETECT" and name in self.logged_names:
            if (now - self.logged_names[name]).total_seconds() < cooldown:
                return False

        self.logged_names[name] = now

        image_path = ""
        if face_bgr is not None and face_bgr.size > 0:
            image_path = self._save_face_snapshot(face_bgr, name)

        if department is None:
            department = self.departments.get(name, "")

        try:
            db = next(get_db())
            log_entry = AccessLog(
                name=name, status=status, confidence=confidence,
                image_path=image_path, timestamp=now,
                check_type=check_type, department=department,
                latitude=latitude, longitude=longitude
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            print(f"[ERROR] log_access DB: {e}")

        activity_item = {
            "name": name, "status": status,
            "confidence": round(confidence, 1),
            "time": now.strftime("%H:%M:%S"),
            "image": image_path
        }
        with self.activity_lock:
            self.recent_activity.insert(0, activity_item)
            self.recent_activity = self.recent_activity[:20]

        if name != "Unknown":
            from backend.core.notify import send_line_notify
            full_path = os.path.join(CAPTURED_DIR, image_path) if image_path else None
            msg = f"🔔 AIFACE: {name} ({confidence:.0f}%) - {status}"
            threading.Thread(target=send_line_notify, args=(msg, full_path), daemon=True).start()

        return True

    def get_recent_activity(self):
        with self.activity_lock:
            return list(self.recent_activity)

detector = FaceDetector()


def _placeholder_frame(message, sub_message=""):
    """A black 640x480 frame carrying a status message, used whenever the
    camera is unavailable so the dashboard shows *why* instead of hanging."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (24, 20, 18)
    cv2.putText(img, "AIFACE", (240, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 110, 100), 2)
    (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, message, ((640 - tw) // 2, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 240), 2)
    if sub_message:
        (sw, _), _ = cv2.getTextSize(sub_message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, sub_message, ((640 - sw) // 2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
    return img


def draw_face_box(frame, box, name, granted, confidence=0.0):
    """Return a copy of `frame` with a rectangle (and name tag) around the
    detected face — green when granted, red when denied. No face, no box."""
    out = frame.copy()
    if box is None:
        return out
    color = (94, 197, 34) if granted else (68, 68, 239)  # BGR
    x1, y1, x2, y2 = box
    thick = max(2, out.shape[1] // 200)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    # Corner accents for a "scanner" look
    L = max(12, (x2 - x1) // 5)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(out, (cx, cy), (cx + dx * L, cy), color, thick * 2)
        cv2.line(out, (cx, cy), (cx, cy + dy * L), color, thick * 2)
    # cv2.putText can't render Thai (shows "??"), so the tag is ASCII-only;
    # the real name is shown in the dashboard banner instead.
    label = f"MATCH {confidence:.0f}%" if granted else "UNKNOWN"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = y1 - 8 if y1 - th - 14 > 0 else y2 + th + 8
    cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 10, ty + 6), color, -1)
    cv2.putText(out, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class CameraStream:
    """Background camera reader that opens lazily and reconnects on its own.

    Opening the device at import time meant a missing/busy webcam broke the
    whole server; now the app always starts and the camera is retried instead.
    """

    RETRY_SECONDS = 3.0

    # OpenCV's DSHOW/MSMF backends are not safe to open/release from several
    # threads at once — doing so (e.g. /camera/list probing while the loop
    # reconnects) corrupted the heap and killed the whole server.
    capture_lock = threading.RLock()

    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.running = False
        self.frame = None
        self.frame_id = 0  # bumped on every new capture, so readers can skip duplicates
        self.last_error = "Camera not started"
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.thread = None

    # ── lifecycle ──

    def start(self):
        with self.state_lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        print(f"[INFO] Camera stream starting (index {self.src})...")

    def stop(self):
        with self.state_lock:
            self.running = False
            thread = self.thread
            self.thread = None
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        # If the old thread is still stuck in cap.read(), releasing here would
        # free the device under it; the loop releases it itself on exit.
        if not (thread and thread.is_alive()):
            self._release()
        with self.frame_lock:
            self.frame = None

    def restart(self, src=None):
        """Stop, optionally switch device index, and start again."""
        self.stop()
        if src is not None:
            self.src = int(src)
        self.last_error = "Reconnecting camera..."
        self.start()

    def _release(self):
        with self.capture_lock:
            self._release_unlocked()

    def _release_unlocked(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # ── capture loop ──

    def _open(self):
        """Try each Windows capture backend in turn; return an open cap or None."""
        backends = []
        for name in ("CAP_DSHOW", "CAP_MSMF"):
            if hasattr(cv2, name):
                backends.append(getattr(cv2, name))
        backends.append(None)

        with self.capture_lock:
            return self._open_unlocked(backends)

    def _open_unlocked(self, backends):
        for backend in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(self.src) if backend is None else cv2.VideoCapture(self.src, backend)
                if cap.isOpened():
                    # MJPG gives USB webcams a smooth, low-bandwidth stream.
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return cap
            except Exception:
                pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        return None

    def _loop(self):
        fail_count = 0
        me = threading.current_thread()
        # A restart starts a fresh thread; an old one that outlived stop()'s
        # join must exit instead of running alongside it.
        while self.running and self.thread is me:
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    self.last_error = f"Camera {self.src} not found or in use by another app"
                    time.sleep(self.RETRY_SECONDS)
                    continue
                self.last_error = ""
                fail_count = 0
                print(f"[INFO] Camera {self.src} opened.")

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                ret, frame = False, None
                self.last_error = str(e)

            if ret and frame is not None and frame.size > 0:
                fail_count = 0
                with self.frame_lock:
                    self.frame = frame
                    self.frame_id += 1
            else:
                fail_count += 1
                # A few dropped reads are normal; a sustained run means the
                # device went away, so drop it and let the loop reconnect.
                if fail_count > 30:
                    print(f"[WARN] Camera {self.src} stopped responding; reconnecting...")
                    self._release()
                    with self.frame_lock:
                        self.frame = None
                    self.last_error = "Camera disconnected, reconnecting..."
                    fail_count = 0
                    time.sleep(self.RETRY_SECONDS)
                else:
                    time.sleep(0.01)

        # Don't pull the device out from under a newer thread that took over.
        if self.thread is None or self.thread is me:
            self._release()

    # ── consumers ──

    def read(self):
        ret, frame, _ = self.read_with_id()
        return ret, frame

    def read_with_id(self):
        with self.frame_lock:
            if self.frame is not None:
                return True, self.frame.copy(), self.frame_id
            return False, None, self.frame_id

    def status(self):
        with self.frame_lock:
            has_frame = self.frame is not None
        return {
            "running": self.running,
            "index": self.src,
            "connected": has_frame and not self.last_error,
            "error": self.last_error,
        }


camera = CameraStream(src=int(config_store.get("camera_index", 0)))


def list_cameras(max_index=6):
    """Probe device indexes 0..max_index-1 and return the ones that deliver a
    frame (built-in webcams and USB webcams alike).

    The camera currently streaming is already held open, so it can't be probed
    a second time — it's reported as available without opening it.
    """
    found = []
    for idx in range(max_index):
        if camera.running and idx == camera.src:
            found.append(idx)
            continue
        with CameraStream.capture_lock:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        found.append(idx)
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
    return found


def generate_frames():
    """MJPEG generator for /api/video_feed.

    This is a plain, unprocessed camera preview — no detection or recognition
    runs here at all. The AI only looks at a frame when the entry-scan button
    is pressed (POST /api/scan -> detector.identify_once) or when the auto-scan
    sees a face appear (POST /api/scan?auto=1, gated by face_present).
    """
    camera.start()
    last_frame_id = -1
    last_sent = 0.0

    while True:
        ret, frame, frame_id = camera.read_with_id()

        # Only ever send a frame the camera actually produced — re-encoding the
        # cached one in a tight loop burned CPU and bandwidth for nothing.
        if ret and frame_id == last_frame_id:
            time.sleep(0.005)
            continue
        if ret and preview_idle and time.time() - last_sent < 1.0 / IDLE_PREVIEW_FPS:
            time.sleep(0.01)
            continue
        last_frame_id = frame_id
        last_sent = time.time()

        if not ret:
            status = camera.status()
            placeholder = _placeholder_frame(
                "Waiting for camera...",
                status.get("error") or f"Camera index {status.get('index')}"
            )
            ok, buffer = cv2.imencode('.jpg', placeholder)
            if ok:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
