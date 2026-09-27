"""
face_unlock — face recognition enrollment & unlock plugin.

Drop-in JARVIS plugin (see plugins/_template.py). Nothing else in the project
was changed to add this: it is discovered automatically at startup exactly
like every other file in plugins/, the same way plugins/greet_vision.py is.

What it does
------------
Lets the user enroll their face once, then unlock JARVIS with it afterwards.

  * "enroll my face" / "set up face unlock" / "register my face"
        Opens the webcam for a few seconds, captures several views of the
        user's face, and stores a face profile locally in
        memory/face_profile.dat (never leaves the machine, never uploaded
        anywhere).

  * "unlock with my face" / "unlock jarvis with face recognition" /
    "face unlock"
        Opens the webcam, checks whoever is in frame against the stored
        profile, and reports whether it's a match. On a match it writes
        memory/face_unlock_state.json with {"unlocked": true}. On failure it
        writes {"unlocked": false} and says access is denied.

  * "lock jarvis" — flips the stored state back to locked without touching
    the enrolled face data.

  * "is my face enrolled" / "face unlock status" — reports what's enrolled
    and the current locked/unlocked state.

  * "remove my face data" / "delete my face enrollment" — deletes the
    stored profile.

Recognition engine
-------------------
If the optional `face_recognition` package (dlib-based, 128-d face
embeddings — the same kind of engine used by most open-source face-unlock
projects) is installed, it's used automatically for accurate recognition.
If it isn't installed, this plugin falls back to a lightweight recognizer
built from OpenCV's bundled Haar cascade plus a local-binary-pattern
histogram and a pixel-correlation check — no extra install required, exactly
like the hand-detection fallback in greet_vision.py. Console output is
prefixed "[FaceUnlock]" the same way, for troubleshooting.

Scope & honesty about what "unlock" means
------------------------------------------
This unlocks *JARVIS* (gates this plugin's own state, which other automations
can check via `is_unlocked()` below). It does NOT and cannot type your
Windows/macOS login password into the OS lock screen — modern operating
systems deliberately block synthetic keyboard/mouse input on the secure
login screen so that exactly this kind of trick isn't possible. For actual
Windows-lock-screen face unlock, use Windows Hello (Settings → Accounts →
Sign-in options), which has the hardware/OS-level access this plugin can't
get from a regular Python script.

Security note: this is a convenience feature, not a hardened biometric
security system — the lightweight fallback recognizer in particular can be
fooled by a photo of the enrolled user. Install `face_recognition` for
meaningfully better accuracy, and don't rely on this alone to protect
anything sensitive.
"""
from __future__ import annotations

import json
import pickle
import platform
import sys
import time
from pathlib import Path

PLUGIN = {
    "name": "face_unlock",
    "description": (
        "Enrolls the user's face and then lets them unlock JARVIS by looking "
        "at the webcam, using face recognition. Use this whenever the user "
        "asks to 'enroll my face', 'set up face unlock', 'register my face', "
        "'unlock with my face', 'unlock jarvis with face recognition', "
        "'face unlock', 'lock jarvis', 'am I enrolled', 'face unlock status', "
        "or 'remove/delete my face data'. Do NOT use this for the "
        "greet_vision tool's 'say hi when you see me' request — that only "
        "detects presence, it doesn't verify identity or unlock anything."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": (
                    "'enroll' (capture and store the user's face), 'unlock' "
                    "(verify the face in view against the stored profile and "
                    "unlock — default), 'lock' (mark JARVIS locked again), "
                    "'status' (report enrollment/lock state), or 'reset' / "
                    "'remove' (delete the stored face profile)."
                ),
            },
            "timeout_seconds": {
                "type": "NUMBER",
                "description": (
                    "How long to watch the camera before giving up, in "
                    "seconds (default 10, max 25). Applies to 'enroll' and "
                    "'unlock'."
                ),
            },
        },
        "required": [],
    },
}


# ── Paths & config (self-contained — mirrors greet_vision.py) ───────────────

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def _memory_dir() -> Path:
    return _base_dir() / "memory"


def _profile_path() -> Path:
    return _memory_dir() / "face_profile.dat"


def _state_path() -> Path:
    return _memory_dir() / "face_unlock_state.json"


def _config_path() -> Path:
    return _base_dir() / "config" / "api_keys.json"


def _load_config() -> dict:
    try:
        return json.loads(_config_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_os() -> str:
    cfg = _load_config()
    val = str(cfg.get("os_system", "")).strip().lower()
    if val in ("windows", "mac", "linux"):
        return val
    sysname = platform.system()
    if sysname == "Darwin":
        return "mac"
    if sysname == "Linux":
        return "linux"
    return "windows"


def _cv2_backend(cv2):
    os_name = _get_os()
    if os_name == "windows":
        return cv2.CAP_DSHOW
    if os_name == "mac":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _camera_index() -> int:
    cfg = _load_config()
    try:
        return int(cfg.get("camera_index", 0))
    except Exception:
        return 0


def _open_camera(cv2):
    """Same probing strategy as greet_vision.py's _open_camera."""
    configured = _camera_index()
    backends = [(_cv2_backend(cv2), "preferred")]
    if _cv2_backend(cv2) != cv2.CAP_ANY:
        backends.append((cv2.CAP_ANY, "CAP_ANY fallback"))

    tried = []
    for backend, backend_label in backends:
        for idx in [configured] + [i for i in range(6) if i != configured]:
            if (backend, idx) in tried:
                continue
            tried.append((backend, idx))
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok = False
                for _ in range(8):
                    ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[FaceUnlock] Camera opened: index={idx}, backend={backend_label}")
                    return cap
            cap.release()

    print(f"[FaceUnlock] No usable camera found. Tried indices/backends: {tried}")
    return None


# ── State (locked/unlocked) ──────────────────────────────────────────────────

def _save_state(unlocked: bool) -> None:
    try:
        _memory_dir().mkdir(parents=True, exist_ok=True)
        _state_path().write_text(
            json.dumps({"unlocked": bool(unlocked), "updated_at": time.time()}),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[FaceUnlock] ⚠️ could not write state file: {e}")


def _load_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_unlocked() -> bool:
    """Convenience for other code: True if the last face-unlock check
    succeeded. Nothing in the project calls this today — it's exposed so a
    future gate can check it without this plugin (or anything else) needing
    to change."""
    return bool(_load_state().get("unlocked", False))


# ── Face profile storage ─────────────────────────────────────────────────────

def _save_profile(method: str, samples: list) -> None:
    _memory_dir().mkdir(parents=True, exist_ok=True)
    with open(_profile_path(), "wb") as f:
        pickle.dump({"method": method, "samples": samples, "enrolled_at": time.time()}, f)


def _load_profile():
    p = _profile_path()
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[FaceUnlock] ⚠️ could not read stored face profile: {e}")
        return None


# ── Face detection (OpenCV Haar cascade — same source greet_vision uses) ────

_face_cascade = None

def _get_face_cascade(cv2):
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
        if _face_cascade.empty():
            print(f"[FaceUnlock] ⚠️ Could not load face cascade from: {path}")
    return _face_cascade


def _detect_face_box(cv2, frame):
    """Returns the largest detected face as (x, y, w, h), or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = _get_face_cascade(cv2)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(faces[0])


# ── Optional deep-learning engine: face_recognition (dlib 128-d embeddings) ─

_fr_module = None
_fr_available = None

def _get_face_recognition():
    global _fr_module, _fr_available
    if _fr_available is None:
        try:
            import face_recognition
            _fr_module = face_recognition
            _fr_available = True
        except Exception:
            _fr_module = None
            _fr_available = False
    return _fr_module if _fr_available else None


def _encode_face_fr(face_recognition, rgb_frame):
    locations = face_recognition.face_locations(rgb_frame)
    if not locations:
        return None

    def _area(loc):
        top, right, bottom, left = loc
        return max(0, bottom - top) * max(0, right - left)

    locations = sorted(locations, key=_area, reverse=True)
    encodings = face_recognition.face_encodings(rgb_frame, [locations[0]])
    if not encodings:
        return None
    return encodings[0]


# ── Fallback lightweight recognizer (LBP histogram + pixel correlation) ─────
# Used automatically when `face_recognition` isn't installed, so face_unlock
# still works with zero extra packages beyond what the project already needs
# (opencv-python + numpy). Less accurate than a deep model — see the module
# docstring's security note.

def _face_roi_gray(cv2, frame, box, size=200):
    x, y, w, h = box
    roi = frame[max(0, y):y + h, max(0, x):x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size))
    gray = cv2.equalizeHist(gray)
    return gray


def _lbp_histogram(gray):
    import numpy as np
    img = gray.astype(np.int16)
    center = img[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = img[1 + dy:1 + dy + center.shape[0], 1 + dx:1 + dx + center.shape[1]]
        code = code | ((neighbor >= center).astype(np.uint8) << i)
    hist, _ = np.histogram(code, bins=256, range=(0, 256))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist


def _thumbnail(cv2, gray, size=48):
    return cv2.resize(gray, (size, size)).astype("float32")


def _hist_similarity(h1, h2) -> float:
    import numpy as np
    a = h1 - h1.mean()
    b = h2 - h2.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / denom)


def _pixel_similarity(t1, t2) -> float:
    import numpy as np
    a = (t1 - t1.mean()).flatten()
    b = (t2 - t2.mean()).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / denom)


# ── Enrollment ────────────────────────────────────────────────────────────────

def _enroll(cv2, cap, seconds: float, need_samples: int, log) -> tuple[str, list]:
    fr = _get_face_recognition()
    method = "face_recognition" if fr else "lbp"
    samples: list = []
    start = time.time()
    last_capture = 0.0

    while time.time() - start < seconds and len(samples) < need_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_capture < 0.35:   # spread samples out a bit
            time.sleep(0.02)
            continue

        try:
            if method == "face_recognition":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                enc = _encode_face_fr(fr, rgb)
                if enc is not None:
                    samples.append(enc)
                    last_capture = now
                    log(f"JARVIS: Captured sample {len(samples)}/{need_samples}...")
            else:
                box = _detect_face_box(cv2, frame)
                if box is not None:
                    gray = _face_roi_gray(cv2, frame, box)
                    hist = _lbp_histogram(gray)
                    thumb = _thumbnail(cv2, gray)
                    samples.append((hist, thumb))
                    last_capture = now
                    log(f"JARVIS: Captured sample {len(samples)}/{need_samples}...")
        except Exception as e:
            print(f"[FaceUnlock] ⚠️ enrollment frame error: {e}")

        time.sleep(0.03)

    return method, samples


# ── Verification ──────────────────────────────────────────────────────────────

def _verify_frame(cv2, frame, profile):
    """Returns ('distance', value) for face_recognition (lower=better) or
    ('score', value) for the lbp fallback (higher=better), or None if no
    face was found in this frame."""
    method = profile.get("method")
    samples = profile.get("samples") or []
    if not samples:
        return None

    if method == "face_recognition":
        fr = _get_face_recognition()
        if fr is None:
            return None
        import numpy as np
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enc = _encode_face_fr(fr, rgb)
        if enc is None:
            return None
        distances = fr.face_distance(samples, enc)
        return ("distance", float(np.min(distances)))

    box = _detect_face_box(cv2, frame)
    if box is None:
        return None
    gray = _face_roi_gray(cv2, frame, box)
    hist = _lbp_histogram(gray)
    thumb = _thumbnail(cv2, gray)
    best = -1.0
    for s_hist, s_thumb in samples:
        score = 0.5 * _hist_similarity(hist, s_hist) + 0.5 * _pixel_similarity(thumb, s_thumb)
        best = max(best, score)
    return ("score", best)


# ── Plugin entry point ───────────────────────────────────────────────────────

def run(parameters: dict, player=None, session_memory=None) -> str:
    action = str(parameters.get("action", "unlock") or "unlock").strip().lower()
    if action not in ("enroll", "unlock", "lock", "status", "reset", "remove"):
        action = "unlock"

    try:
        timeout = float(parameters.get("timeout_seconds", 10) or 10)
    except (TypeError, ValueError):
        timeout = 10.0
    timeout = max(3.0, min(timeout, 25.0))

    def _log(msg: str) -> None:
        if player:
            try:
                player.write_log(msg)
            except Exception:
                pass

    print(f"[FaceUnlock] run() called — action={action}, timeout={timeout}s")

    # ── actions that don't need the camera ──
    if action == "status":
        profile = _load_profile()
        state = _load_state()
        if not profile:
            return "Sir, no face is enrolled yet. Say 'enroll my face' to set up face unlock."
        method = profile.get("method")
        n = len(profile.get("samples", []))
        engine = "the face_recognition deep-learning model" if method == "face_recognition" else "the built-in lightweight recognizer"
        unlocked = state.get("unlocked", False)
        return (f"Sir, a face is enrolled ({n} samples, using {engine}). "
                f"JARVIS is currently {'unlocked' if unlocked else 'locked'}.")

    if action in ("reset", "remove"):
        try:
            if _profile_path().exists():
                _profile_path().unlink()
            _save_state(False)
            return "Sir, the enrolled face profile has been deleted."
        except Exception as e:
            return f"Sir, I couldn't remove the face profile: {e}"

    if action == "lock":
        _save_state(False)
        return "JARVIS is now locked. Face verification will be required to unlock again."

    # ── actions that need the camera: enroll, unlock ──
    try:
        import cv2
        print(f"[FaceUnlock] OpenCV {cv2.__version__} OK")
    except ImportError as e:
        print(f"[FaceUnlock] ⚠️ OpenCV import failed: {e}")
        return "Sir, I can't see the camera — OpenCV isn't installed. Run: pip install opencv-python"

    cap = _open_camera(cv2)
    if cap is None:
        return ("Sir, I couldn't open the webcam on any camera index. Check it's "
                "connected, not in use by another app, and that JARVIS has camera "
                "permission in your system's privacy settings.")

    fr_available = _get_face_recognition() is not None
    print(f"[FaceUnlock] Recognition engine: {'face_recognition (dlib)' if fr_available else 'lightweight LBP fallback (install face_recognition for better accuracy)'}")

    try:
        if action == "enroll":
            _log("JARVIS: Look at the camera — enrolling your face now. Turn your head slightly, side to side.")
            if not fr_available:
                _log("JARVIS: face_recognition isn't installed — using the built-in lightweight recognizer. "
                     "For stronger accuracy: pip install face_recognition")

            method, samples = _enroll(cv2, cap, seconds=min(timeout, 12.0) if timeout != 10.0 else 12.0,
                                       need_samples=8, log=_log)

            print(f"[FaceUnlock] enrollment done — method={method}, samples={len(samples)}")

            if len(samples) < 3:
                return ("Sir, I couldn't capture enough clear views of your face. "
                        "Make sure you're facing the camera in good lighting, then say "
                        "'enroll my face' again.")

            _save_profile(method, samples)
            _save_state(True)
            engine = "deep-learning face recognition" if method == "face_recognition" else "the built-in lightweight recognizer"
            msg = f"Face enrolled successfully using {engine}, sir. Say 'unlock with my face' any time to unlock JARVIS."
            _log(f"JARVIS: {msg}")
            return msg

        # action == "unlock"
        profile = _load_profile()
        if not profile:
            return "Sir, no face is enrolled yet. Say 'enroll my face' first."

        if profile.get("method") == "face_recognition" and not fr_available:
            return ("Sir, this profile was enrolled with the face_recognition library, "
                    "which is no longer installed. Re-enroll, or reinstall it with: "
                    "pip install face_recognition")

        _log(f"JARVIS: Checking your face against the enrolled profile (up to {int(timeout)}s)...")

        cfg = _load_config()
        distance_tolerance = float(cfg.get("face_unlock_tolerance", 0.6))
        score_threshold = float(cfg.get("face_unlock_threshold", 0.55))

        consecutive_matches = 0
        checks = 0
        need_consecutive = 3
        start = time.time()

        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            try:
                result = _verify_frame(cv2, frame, profile)
            except Exception as e:
                print(f"[FaceUnlock] ⚠️ verification frame error: {e}")
                result = None

            if result is not None:
                kind, value = result
                checks += 1
                is_match = (value <= distance_tolerance) if kind == "distance" else (value >= score_threshold)
                consecutive_matches = consecutive_matches + 1 if is_match else max(0, consecutive_matches - 1)
                if checks % 10 == 0:
                    print(f"[FaceUnlock] checks={checks}, last={kind}={value:.3f}, consecutive_matches={consecutive_matches}")
                if consecutive_matches >= need_consecutive:
                    break

            time.sleep(0.03)

        print(f"[FaceUnlock] unlock attempt done — checks={checks}, consecutive_matches={consecutive_matches}")

        if consecutive_matches >= need_consecutive:
            _save_state(True)
            msg = "Face recognized. Welcome back, sir — JARVIS is unlocked."
            _log(f"JARVIS: {msg}")
            return msg

        _save_state(False)
        if checks == 0:
            msg = ("Sir, I couldn't get a clear look at a face during the check. "
                   "Face the camera in good lighting and try again.")
        else:
            msg = "Sir, I don't recognize that face. Access denied."
        _log(f"JARVIS: {msg}")
        return msg

    finally:
        cap.release()
