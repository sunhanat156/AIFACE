from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Request
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database.models import get_db, User, AccessLog
from backend.core.ai_processor import (
    generate_frames, detector, camera, CAPTURED_DIR, list_cameras, draw_face_box,
    face_present, scene_changed,
    extract_multi_encodings, extract_multi_encodings_from_files
)
from backend.core import config_store
from backend.core import ai_processor
from fastapi.responses import Response
import numpy as np
import cv2
import base64
import json
import os
import datetime
import threading
import time
from typing import List

router = APIRouter()


# ═══════════════ Video ═══════════════

@router.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/health")
def health():
    """Used by the launcher to know when the server is ready to be opened."""
    return {
        "status": "ok",
        "known_faces": len(set(detector.known_face_names)),
        "face_vectors": len(detector.known_face_names),
    }


# ═══════════════ Camera ═══════════════

@router.get("/camera/status")
def camera_status():
    return camera.status()


@router.post("/camera/restart")
def camera_restart():
    """Reconnect the webcam — e.g. after plugging it back in or freeing it
    from another app — without restarting the whole server."""
    camera.restart(int(config_store.get("camera_index", 0)))
    return {"status": "success", "camera": camera.status()}


@router.get("/camera/list")
def camera_list():
    """Cameras currently available (built-in + USB webcams) and the selected one."""
    try:
        # DirectShow order matches the OpenCV CAP_DSHOW indexes.
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
    except Exception:
        names = []
    return {
        "cameras": [
            {"index": i, "name": names[i] if i < len(names) else f"Camera {i}"}
            for i in list_cameras()
        ],
        "current": camera.src,
    }


@router.post("/camera/select")
async def camera_select(request: Request):
    """Switch to another camera by device index, save it, and reconnect."""
    data = await request.json()
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="index must be an integer")
    if index < 0:
        raise HTTPException(status_code=400, detail="index must be >= 0")
    config_store.save_config({"camera_index": index})
    camera.restart(index)
    return {"status": "success", "camera": camera.status()}


# Auto-scan state. A decision needs the same identity on several consecutive
# fresh frames (one odd frame of a known face at an angle often reads as
# Unknown), and a person who stays in view is logged once per visit: every
# confirmed sighting slides their cooldown forward, so they're only logged
# again after being gone for the whole cooldown.
#
# The GPU face engine only runs while a face is in view (scene_changed ->
# face_present gates, both CPU), and only until that visit has a decision —
# after that it rests until the face has been gone for AUTO_ABSENT_RESET
# seconds, so someone standing in front of the camera isn't re-scanned.
AUTO_CONFIRM_KNOWN = 2
AUTO_CONFIRM_UNKNOWN = 3
AUTO_ABSENT_RESET = 1.5  # s without a face before the next visit can be scanned
_auto_lock = threading.Lock()
_auto = {"frame_id": -1, "name": None, "streak": 0, "present": {},
         "seen_at": 0.0, "done": False, "in_view": False}


def _auto_idle(frame_id, now):
    """Nobody in view: the GPU engine rests and the preview slows down. Once
    they've been gone long enough, the visit is over."""
    _auto["frame_id"], _auto["in_view"] = frame_id, False
    ai_processor.preview_idle = True
    if now - _auto["seen_at"] > AUTO_ABSENT_RESET:
        _auto["name"], _auto["streak"], _auto["done"] = None, 0, False
    return {"status": "idle", "reason": "no_face"}


def _auto_should_log(name, frame_id):
    """Feed one quality-passed detection into the auto-scan state; return a
    skip reason, or None if this sighting should be logged now."""
    if frame_id == _auto["frame_id"]:
        return "stale_frame"
    _auto["frame_id"] = frame_id

    if name == _auto["name"]:
        _auto["streak"] += 1
    else:
        _auto["name"], _auto["streak"] = name, 1
    needed = AUTO_CONFIRM_UNKNOWN if name == "Unknown" else AUTO_CONFIRM_KNOWN
    if _auto["streak"] < needed:
        return "confirming"

    now = time.time()
    cooldown = detector.unknown_cooldown if name == "Unknown" else detector.log_cooldown
    last = _auto["present"].get(name)
    _auto["present"][name] = now
    if last is not None and now - last < cooldown:
        return "already_logged"
    return None


@router.post("/scan")
def scan_entry(auto: bool = False):
    """Room-entry scan: grab the current camera frame, identify whoever is
    in it (quality-gated), and log the attempt. Called by the scan button,
    or with ?auto=1 by the dashboard's continuous auto-scan loop.

    This is the ONLY path that grants/denies entry — the live dashboard feed
    just draws boxes for preview and never logs on its own, so a single bad
    frame from the continuous stream can't accidentally let a stranger in.
    """
    camera.start()
    ok, frame, frame_id = camera.read_with_id()
    if not ok or frame is None:
        raise HTTPException(status_code=503, detail=camera.status().get("error") or "Camera not ready")

    granted = False
    if auto:
        # One lock for decide + log, so two open dashboards can't both log
        # the same sighting.
        with _auto_lock:
            if frame_id == _auto["frame_id"]:
                return {"status": "skipped", "reason": "stale_frame"}
            now = time.time()
            moved = scene_changed(frame)
            if not moved and not _auto["in_view"]:
                # Empty, static scene: no model runs at all.
                return _auto_idle(frame_id, now)
            if moved or not _auto["done"]:
                _auto["in_view"] = face_present(frame)  # CPU only
                if not _auto["in_view"]:
                    return _auto_idle(frame_id, now)
            # (No motion + visit already decided = same person still standing
            # there, so even the CPU check is skipped.)
            ai_processor.preview_idle = False
            _auto["seen_at"] = now
            if _auto["done"]:
                # Same visit, already decided: keep their cooldown sliding
                # (as if still being sighted) without running the engine.
                _auto["frame_id"] = frame_id
                if _auto["name"] is not None:
                    _auto["present"][_auto["name"]] = now
                return {"status": "skipped", "reason": "already_scanned"}
            # A face is in view and undecided: only now does the GPU engine run.
            name, confidence, face_crop, box = detector.identify_once(frame)
            if face_crop is None:
                # Face too small/blurry/far to trust: any half-built streak is void.
                _auto["frame_id"], _auto["name"], _auto["streak"] = frame_id, None, 0
                return {"status": "skipped", "reason": "low_quality"}
            reason = _auto_should_log(name, frame_id)
            if reason == "already_logged":
                _auto["done"] = True
            if reason:
                return {"status": "skipped", "reason": reason}
            _auto["done"] = True
            granted = name != "Unknown"
            status = "Access Granted" if granted else "Access Denied"
            detector.log_access(name, status, confidence, face_crop, check_type="ENTRY")
    else:
        name, confidence, face_crop, box = detector.identify_once(frame)
        granted = name != "Unknown"
        status = "Access Granted" if granted else "Access Denied"
        detector.log_access(name, status, confidence, face_crop, check_type="ENTRY")

    # Frozen frame with a rectangle around the detected face, shown on the dashboard.
    annotated = draw_face_box(frame, box, name, granted, confidence)
    encoded, buffer = cv2.imencode(".jpg", annotated)
    image = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode() if encoded else ""

    return {
        "status": "success",
        "granted": granted,
        "name": name,
        "confidence": round(confidence, 1),
        "face_found": box is not None,
        "image": image,
    }


@router.get("/snapshot")
def snapshot():
    """Return the current raw camera frame as a JPEG.

    Lets the browser enrol a face straight from the live camera, so a first
    run needs no photo files at all.
    """
    camera.start()
    ok, frame = camera.read()
    if not ok or frame is None:
        raise HTTPException(status_code=503, detail=camera.status().get("error") or "Camera not ready")
    encoded, buffer = cv2.imencode(".jpg", frame)
    if not encoded:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


# ═══════════════ Activity ═══════════════

@router.get("/activity")
def get_activity():
    return detector.get_recent_activity()


# ═══════════════ Logs (with delete) ═══════════════

@router.get("/logs")
def get_logs(department: str = "", name: str = "", date_from: str = "", date_to: str = "",
             db: Session = Depends(get_db)):
    """Latest logs, optionally filtered by department, name (partial,
    case-insensitive) and an inclusive YYYY-MM-DD date range."""
    query = db.query(AccessLog)
    if department:
        query = query.filter(AccessLog.department == department)
    if name.strip():
        query = query.filter(AccessLog.name.ilike(f"%{name.strip()}%"))
    try:
        if date_from:
            start = datetime.datetime.strptime(date_from, "%Y-%m-%d")
            query = query.filter(AccessLog.timestamp >= start)
        if date_to:
            end = datetime.datetime.strptime(date_to, "%Y-%m-%d") + datetime.timedelta(days=1)
            query = query.filter(AccessLog.timestamp < end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    searching = bool(name.strip() or date_from or date_to)
    logs = query.order_by(AccessLog.timestamp.desc()).limit(1000 if searching else 200).all()
    return [
        {
            "id": l.id, "name": l.name, "status": l.status,
            "confidence": l.confidence,
            "timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": l.image_path or "",
            "department": l.department or "",
            "check_type": l.check_type or "DETECT",
            "latitude": l.latitude,
            "longitude": l.longitude,
        }
        for l in logs
    ]


@router.delete("/logs/{log_id}")
def delete_log(log_id: int, db: Session = Depends(get_db)):
    """Delete a single log entry and its captured image."""
    log = db.query(AccessLog).filter(AccessLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    # Remove captured image file
    if log.image_path:
        filepath = os.path.join(CAPTURED_DIR, log.image_path)
        if os.path.exists(filepath):
            os.remove(filepath)
    db.delete(log)
    db.commit()
    return {"status": "success"}


@router.delete("/logs/bulk/date/{date_str}")
def delete_logs_by_date(date_str: str, db: Session = Depends(get_db)):
    """Delete all logs for a specific date (YYYY-MM-DD). Removes captured images too."""
    try:
        target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    next_date = target_date + datetime.timedelta(days=1)
    logs = db.query(AccessLog).filter(
        AccessLog.timestamp >= datetime.datetime.combine(target_date, datetime.time.min),
        AccessLog.timestamp < datetime.datetime.combine(next_date, datetime.time.min)
    ).all()

    count = 0
    for log in logs:
        if log.image_path:
            filepath = os.path.join(CAPTURED_DIR, log.image_path)
            if os.path.exists(filepath):
                os.remove(filepath)
        db.delete(log)
        count += 1

    db.commit()
    return {"status": "success", "deleted": count}


@router.delete("/logs/bulk/unknown")
def delete_unknown_logs(db: Session = Depends(get_db)):
    """Delete all 'Unknown' (mismatched) logs and their images."""
    logs = db.query(AccessLog).filter(AccessLog.name == "Unknown").all()
    count = 0
    for log in logs:
        if log.image_path:
            filepath = os.path.join(CAPTURED_DIR, log.image_path)
            if os.path.exists(filepath):
                os.remove(filepath)
        db.delete(log)
        count += 1
    db.commit()
    return {"status": "success", "deleted": count}


# ═══════════════ Captured Images ═══════════════

@router.get("/captured/{filename}")
def get_captured_image(filename: str):
    filepath = os.path.join(CAPTURED_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath, media_type="image/jpeg")


# ═══════════════ Users ═══════════════

@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "id": u.id, "name": u.name,
            "department": u.department or "",
            "registered_at": u.registered_at.strftime("%Y-%m-%d %H:%M:%S"),
            "encoding_count": len(json.loads(u.face_encodings)) if u.face_encodings else 0
        }
        for u in users
    ]


@router.get("/departments")
def get_departments(db: Session = Depends(get_db)):
    """Distinct department names, for the filter dropdowns on Users/Logs/Check-In."""
    rows = db.query(User.department).filter(User.department.isnot(None), User.department != "").distinct().all()
    return sorted({r[0] for r in rows})


@router.post("/register")
async def register_user(
    name: str,
    department: str = "",
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Register a user with one or more face images.
    Each image is augmented (flipped, rotated, brightness adjusted) to generate
    multiple face encodings for maximum recognition accuracy from all angles.
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    # Read all uploaded images
    image_bytes_list = []
    for f in files:
        data = await f.read()
        if data:
            image_bytes_list.append(data)

    if not image_bytes_list:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    # Extract multi-angle encodings from all images
    all_encodings, valid_images = extract_multi_encodings_from_files(image_bytes_list)

    if not all_encodings:
        raise HTTPException(
            status_code=400,
            detail=f"No face found in any of the {len(image_bytes_list)} uploaded images. Please use clear photos showing the face."
        )

    # Save the first valid image as profile picture
    nparr = np.frombuffer(image_bytes_list[0], np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
    img_filename = f"user_{safe_name}.jpg"
    img_path = os.path.join(CAPTURED_DIR, img_filename)
    if img is not None:
        cv2.imwrite(img_path, img)

    # Convert encodings to serializable lists
    enc_list = [enc.tolist() for enc in all_encodings]

    # Save to database
    user = db.query(User).filter(User.name == name).first()
    if user:
        try:
            existing = json.loads(user.face_encodings) if user.face_encodings else []
        except Exception:
            existing = []
        existing.extend(enc_list)
        user.face_encodings = json.dumps(existing)
        user.image_path = img_path
        if department.strip():
            user.department = department.strip()
    else:
        new_user = User(
            name=name,
            department=department.strip() or None,
            face_encodings=json.dumps(enc_list),
            image_path=img_path
        )
        db.add(new_user)

    db.commit()
    detector.load_known_faces()

    return {
        "status": "success",
        "message": f"Registered '{name}' with {len(enc_list)} face vectors from {valid_images} image(s)."
    }


@router.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    detector.load_known_faces()
    return {"status": "success"}


# ═══════════════ Check-In (self-service attendance) ═══════════════

@router.get("/checkin/status")
def checkin_status(name: str, db: Session = Depends(get_db)):
    """The user's most recent IN/OUT event (any day), so the Check-In page
    can default to the sensible next action."""
    last = (
        db.query(AccessLog)
        .filter(AccessLog.name == name, AccessLog.check_type.in_(["IN", "OUT"]))
        .order_by(AccessLog.timestamp.desc())
        .first()
    )
    return {
        "last_check_type": last.check_type if last else None,
        "last_timestamp": last.timestamp.strftime("%Y-%m-%d %H:%M:%S") if last else None,
        "next_action": "OUT" if (last and last.check_type == "IN") else "IN",
    }


@router.post("/checkin")
async def checkin(
    name: str,
    check_type: str,
    latitude: float = None,
    longitude: float = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Self-service clock in/out: the user picks their own name (login) on the
    Check-In page, the camera confirms it's really them, and the event is
    logged with department + GPS."""
    if check_type not in ("IN", "OUT"):
        raise HTTPException(status_code=400, detail="check_type must be IN or OUT")

    user = db.query(User).filter(User.name == name).first()
    if not user:
        raise HTTPException(status_code=404, detail="Unknown user. Please select your registered name.")

    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read the captured image.")

    ok, matched_name, confidence, face_crop = detector.verify_identity(name, img)

    if not ok:
        # Log the failed attempt under the claimed name so it's auditable,
        # but never write a check-in/out event for a face that didn't match.
        detector.log_access(
            name, "Access Denied", confidence, face_crop,
            check_type="DETECT", department=user.department or ""
        )
        reason = "No face detected. Please look at the camera and try again." if matched_name == "Unknown" and confidence == 0.0 and face_crop is None else \
                 f"Face does not match '{name}' (matched: {matched_name})."
        raise HTTPException(status_code=401, detail=reason)

    detector.log_access(
        matched_name, f"Clock {check_type}", confidence, face_crop,
        check_type=check_type, department=user.department or "",
        latitude=latitude, longitude=longitude
    )

    return {
        "status": "success",
        "name": matched_name,
        "check_type": check_type,
        "confidence": round(confidence, 1),
        "department": user.department or "",
    }


# ═══════════════ Config ═══════════════

@router.get("/config")
def get_config():
    return config_store.load_config()


@router.post("/config")
async def update_config(request: Request):
    """Save settings AND apply them live: tolerance/cooldown go straight to the
    detector, and a changed camera index reconnects the webcam."""
    data = await request.json()
    old_index = int(config_store.get("camera_index", 0))

    config = config_store.save_config(data)
    detector.apply_settings(config)

    camera_changed = int(config.get("camera_index", 0)) != old_index
    if camera_changed:
        camera.restart(int(config["camera_index"]))

    return {"status": "success", "camera_restarted": camera_changed, "config": config}
