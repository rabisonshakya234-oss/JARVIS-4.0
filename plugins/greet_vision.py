"""
greet_vision — face & hand recognition greeting plugin.

Drop-in JARVIS plugin (see plugins/_template.py). Nothing else in the project
was changed to add this: it is discovered automatically at startup exactly
like every other file in plugins/.

What it does
------------
Opens the webcam for a short window and watches for either a human face
(OpenCV's built-in Haar cascade — ships with opencv-python, no extra install)
or a hand (MediaPipe Hands if `pip install mediapipe` has been run, otherwise
a lightweight HSV skin-colour + contour fallback that needs nothing extra).
The moment one of them is seen, JARVIS greets the user with "Hi" and stops
watching. If nothing is seen before the timeout, it says so instead of
hanging forever.

Trigger phrases (for Gemini): "say hi when you see me", "greet me", "watch
the camera and say hello", "recognize my face and say hi", "recognize my
hand and say hi", "wave at me", "look for me", or simply saying "hi" /
"hello" / "hey" to JARVIS.

Troubleshooting: every run prints a [GreetVision] line to the console with
the camera index/backend it used, whether MediaPipe is active, and — every
~2 seconds while it's watching — how many frames it has processed and
whether it saw a face/hand yet. If this still doesn't work, copy that
console output; it says exactly which step failed.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

PLUGIN = {
    "name": "greet_vision",
    "description": (
        "Watches the webcam for a moment and greets the user by saying hi as "
        "soon as it recognizes either a human FACE or a HAND in view. Use this "
        "whenever the user asks JARVIS to greet them, say hi/hello when it "
        "sees them, watch for them and wave, recognize their face or hand — "
        "AND ALSO whenever the user simply greets JARVIS with something like "
        "'hi', 'hello', 'hey Jarvis' (with the camera/vision available), since "
        "that greeting is the trigger to look and say hi back. Do NOT use "
        "this for 'what do you see' / describe-the-room requests — this tool "
        "only detects presence of a face or hand, it does not describe the "
        "scene."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "target": {
                "type": "STRING",
                "description": (
                    "What to look for before greeting: 'face' (face only), "
                    "'hand' (hand only), 'both' (require face AND hand at the "
                    "same time), or 'either' (greet on whichever is seen "
                    "first — default)."
                ),
            },
            "timeout_seconds": {
                "type": "NUMBER",
                "description": "How long to watch before giving up, in seconds (default 12, max 30).",
            },
        },
        "required": [],
    },
}


# ── Small self-contained helpers (no imports from other project modules,
#    so this file has zero dependency on anything that could change) ─────────

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


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
    """Try the configured camera index first, then probe a wider range with
    the OS-preferred backend, then fall back to CAP_ANY (some setups report
    a working device only through the generic backend)."""
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
                for _ in range(8):   # warm up so the first real frame isn't black
                    ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[GreetVision] Camera opened: index={idx}, backend={backend_label}")
                    return cap
            cap.release()

    print(f"[GreetVision] No usable camera found. Tried indices/backends: {tried}")
    return None


# ── Face detection (OpenCV Haar cascade — bundled with opencv-python) ───────

_face_cascade = None
_profile_cascade = None

def _get_face_cascades(cv2):
    global _face_cascade, _profile_cascade
    if _face_cascade is None:
        front_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(front_path)
        if _face_cascade.empty():
            print(f"[GreetVision] ⚠️ Could not load face cascade from: {front_path}")
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        _profile_cascade = cv2.CascadeClassifier(profile_path)
    return _face_cascade, _profile_cascade


def _face_detected(cv2, frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    front, profile = _get_face_cascades(cv2)
    # A bit more permissive than a strict 1.2/5 — real-world webcam distance
    # and lighting means the tighter defaults were missing plenty of faces.
    faces = front.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
    if len(faces) > 0:
        return True
    if profile is not None and not profile.empty():
        faces = profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        if len(faces) > 0:
            return True
    return False


# ── Hand detection: MediaPipe if available, otherwise an HSV skin-blob
#    fallback so the feature still works with nothing extra installed ───────

_mp_hands = None
_mp_available = None

def _get_mediapipe_hands():
    global _mp_hands, _mp_available
    if _mp_available is None:
        try:
            import mediapipe as mp
            _mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            _mp_available = True
        except Exception:
            _mp_hands = None
            _mp_available = False
    return _mp_hands if _mp_available else None


def _hand_detected_mediapipe(cv2, frame) -> bool:
    hands = _get_mediapipe_hands()
    if hands is None:
        return False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    return bool(result.multi_hand_landmarks)


def _hand_detected_skin_fallback(cv2, frame) -> bool:
    """No mediapipe installed: look for a reasonably large skin-coloured blob
    with a hand-like shape. Not gesture-aware, just presence-aware — good
    enough to notice someone holding a hand up / waving at the camera."""
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 60], dtype="uint8")
    upper1 = np.array([20, 150, 255], dtype="uint8")
    lower2 = np.array([170, 30, 60], dtype="uint8")
    upper2 = np.array([180, 150, 255], dtype="uint8")
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    frame_area = frame.shape[0] * frame.shape[1]
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # A raised/waving hand fills a noticeable but not huge chunk of the frame.
    return frame_area * 0.02 < area < frame_area * 0.45


def _hand_detected(cv2, frame) -> bool:
    if _get_mediapipe_hands() is not None:
        return _hand_detected_mediapipe(cv2, frame)
    return _hand_detected_skin_fallback(cv2, frame)


# ── Plugin entry point ───────────────────────────────────────────────────────

def run(parameters: dict, player=None, session_memory=None) -> str:
    target = str(parameters.get("target", "either") or "either").strip().lower()
    if target not in ("face", "hand", "both", "either"):
        target = "either"

    try:
        timeout = float(parameters.get("timeout_seconds", 12) or 12)
    except (TypeError, ValueError):
        timeout = 12.0
    timeout = max(3.0, min(timeout, 30.0))

    def _log(msg: str) -> None:
        if player:
            try:
                player.write_log(msg)
            except Exception:
                pass

    print(f"[GreetVision] run() called — target={target}, timeout={timeout}s")

    try:
        import cv2
        print(f"[GreetVision] OpenCV {cv2.__version__} OK")
    except ImportError as e:
        print(f"[GreetVision] ⚠️ OpenCV import failed: {e}")
        return "Sir, I can't see the camera — OpenCV isn't installed. Run: pip install opencv-python"

    cap = _open_camera(cv2)
    if cap is None:
        return ("Sir, I couldn't open the webcam on any camera index. Check it's "
                "connected, not in use by another app (like the dashboard preview "
                "or Zoom), and that JARVIS has camera permission in Windows privacy "
                "settings.")

    used_mediapipe = _get_mediapipe_hands() is not None
    print(f"[GreetVision] Hand detector: {'MediaPipe' if used_mediapipe else 'skin-color fallback (install mediapipe for better accuracy)'}")
    if target in ("hand", "both", "either") and not used_mediapipe:
        _log("JARVIS: mediapipe not installed — using basic skin-color hand "
             "detection. For more accurate hand tracking: pip install mediapipe")

    _log(f"JARVIS: Watching the camera for a {target} (up to {int(timeout)}s)...")

    saw_face = False
    saw_hand = False
    frames_read = 0
    read_failures = 0
    start = time.time()
    last_progress = start

    try:
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret or frame is None:
                read_failures += 1
                time.sleep(0.05)
                continue
            frames_read += 1

            if target in ("face", "both", "either") and not saw_face:
                try:
                    saw_face = _face_detected(cv2, frame)
                except Exception as e:
                    print(f"[GreetVision] ⚠️ face detection error: {e}")

            if target in ("hand", "both", "either") and not saw_hand:
                try:
                    saw_hand = _hand_detected(cv2, frame)
                except Exception as e:
                    print(f"[GreetVision] ⚠️ hand detection error: {e}")

            now = time.time()
            if now - last_progress >= 2:
                print(f"[GreetVision] {frames_read} frames read ({read_failures} failed reads) — "
                      f"face={saw_face} hand={saw_hand}")
                last_progress = now

            if target == "face" and saw_face:
                break
            if target == "hand" and saw_hand:
                break
            if target == "either" and (saw_face or saw_hand):
                break
            if target == "both" and saw_face and saw_hand:
                break

            time.sleep(0.03)
    finally:
        cap.release()

    print(f"[GreetVision] done — frames_read={frames_read}, read_failures={read_failures}, "
          f"saw_face={saw_face}, saw_hand={saw_hand}")

    if frames_read == 0:
        return ("Sir, the webcam opened but never returned a usable frame — it may be "
                "held by another application. Close anything else using the camera and try again.")

    if target == "face":
        success = saw_face
    elif target == "hand":
        success = saw_hand
    elif target == "both":
        success = saw_face and saw_hand
    else:
        success = saw_face or saw_hand

    if not success:
        msg = (f"I watched the camera for {int(timeout)} seconds but didn't spot a "
               f"{target if target != 'either' else 'face or hand'}. Try moving "
               f"closer to the camera or improving the lighting.")
        _log(f"JARVIS: {msg}")
        return msg

    if saw_face and saw_hand:
        msg = "Hi! 👋 I can see your face and your hand — great to see you, sir."
    elif saw_face:
        msg = "Hi! 👋 I can see your face — good to see you, sir."
    else:
        msg = "Hi! 👋 I can see your hand — hello there, sir."

    _log(f"JARVIS: {msg}")
    return msg
