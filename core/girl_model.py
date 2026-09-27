"""
Girl model plug-in — adds a BOY / GIRL assistant-model switch to JARVIS.

Design goal: **the existing code is not modified.**  Everything the girl model
needs lives in this one file, and it attaches itself to the running app through
`install(ui_module)`, which `ui.py` calls once at its very end.  Delete that one
call and the app behaves exactly as it did before.

What "girl model" means
-----------------------
  * FACE   – a feminised version of the same measured head mesh (narrower jaw,
             slimmer neck, larger eyes, fuller lips, smaller nose, arched
             brows) plus long hair and eyelashes.  Same topology as the boy
             head, so lip-sync / blinking / gaze all work unchanged.
  * VOICE  – switching to Girl picks a female Gemini voice (Kore), switching
             back to Boy restores a male one (Charon).  It goes through the
             app's existing voice-change path, so the Live session rebuilds by
             itself.
  * PERSONA– a short [PERSONA] block is appended to the system prompt so the
             assistant also *talks* as a woman (gendered languages, etc.).

Where it hooks in (all by wrapping, never by editing):
  * HudCanvas.__init__        → the avatar is wrapped in `SwitchableAvatar`
  * CustomizeOverlay.__init__ → a "ASSISTANT MODEL  [BOY] [GIRL]" row is added
  * CustomizeOverlay._save    → the choice is saved (+ voice follows it)
  * CustomizeOverlay._cancel  → a live preview is reverted
  * main._load_system_prompt  → persona text appended when model == girl
The choice is stored in config/api_keys.json under the key "assistant_model".
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Model selection state
# ─────────────────────────────────────────────────────────────────────────────

BOY, GIRL = "boy", "girl"
MODELS = (BOY, GIRL)
CONFIG_KEY = "assistant_model"

# Gemini Live prebuilt voices, by the gender they read as. Kore / Aoede are the
# female voices in memory.config_manager.AVAILABLE_VOICES; the rest are male.
FEMALE_VOICES = ("Kore", "Aoede")
DEFAULT_VOICE = {BOY: "Charon", GIRL: "Kore"}

_active: str | None = None      # in-memory copy, so paint() never touches disk


def _config_file():
    from memory.config_manager import CONFIG_FILE
    return CONFIG_FILE


def get_model() -> str:
    """The saved model ("boy" if nothing has been chosen — today's behaviour)."""
    try:
        from memory.config_manager import load_api_keys
        m = str(load_api_keys().get(CONFIG_KEY, BOY)).lower()
    except Exception:
        m = BOY
    return m if m in MODELS else BOY


def save_model(model: str) -> None:
    """Persist the model, keeping every other key in api_keys.json intact."""
    from memory.config_manager import ensure_config_dir
    model = model if model in MODELS else BOY
    ensure_config_dir()
    path = _config_file()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[CONFIG_KEY] = model
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    set_active(model)


def active() -> str:
    global _active
    if _active is None:
        _active = get_model()
    return _active


def set_active(model: str) -> None:
    """Change what is drawn/spoken right now (also used for live preview)."""
    global _active
    _active = model if model in MODELS else BOY


def voice_fits(model: str, voice: str) -> bool:
    return (voice in FEMALE_VOICES) == (model == GIRL)


# ─────────────────────────────────────────────────────────────────────────────
# Persona text for the system prompt
# ─────────────────────────────────────────────────────────────────────────────

_GIRL_PERSONA = (
    "[PERSONA]\n"
    "You are a woman. Your face, your voice and the way you present yourself "
    "are feminine. Where a language marks the speaker's gender (verb endings, "
    "adjectives, titles), speak of yourself in the feminine. Keep every rule "
    "above unchanged — this only changes who you are, not what you can do.\n"
)


def persona_text() -> str:
    return _GIRL_PERSONA if get_model() == GIRL else ""


# ─────────────────────────────────────────────────────────────────────────────
# Girl head mesh
# ─────────────────────────────────────────────────────────────────────────────
# Built from the boy head (same measured face, same vertex order for the first
# n_head vertices) so every landmark ring, the jaw rig and the lip rig line up.
# Coordinates: +x viewer's right, +y up (crown 1.0, chin -1.0), +z out of face.

_CAM_HEAD_C = np.array([0.0, 0.10, -0.30], dtype=np.float32)   # skull centre
_HAIR_TOP_Y = 0.52          # top of the long-hair sheet (below crown volume)
_HAIR_BOTTOM_Y = -1.28      # stays above the neck's lowest point (-1.32)
_HAIR_ROWS, _HAIR_COLS = 14, 25
_HAIR_THETA = math.radians(126.0)       # how far round to the front it reaches

_GIRL_CACHE: dict | None = None


def _gauss(x, s):
    return np.exp(-(x / s) ** 2)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _feminise(verts: np.ndarray, base: dict) -> np.ndarray:
    """Return a copy of `verts` with feminine proportions (same vertex count)."""
    from core.avatar_mesh import LANDMARKS

    v = verts.copy()
    n_face, n_head = base["n_face"], base["n_head"]
    head = slice(0, n_head)

    # 1. Narrower jaw and a softer, slightly shorter chin. Applied to every head
    #    vertex from its own height, so the cranium sweep stays welded to the face.
    y = v[head, 1]
    jaw_t = _smooth((0.05 - y) / 1.05) ** 1.15          # 0 at the eyes → 1 at chin
    v[head, 0] *= 1.0 - 0.13 * jaw_t
    v[head, 1] = np.where(y < -0.55, y + 0.035 * jaw_t, y)

    # 2. Slimmer neck, about its own axis.
    axis_z = -0.15
    v[n_head:, 0] *= 0.80
    v[n_head:, 2] = axis_z + (v[n_head:, 2] - axis_z) * 0.88

    # 3. Larger, rounder eyes — a gaussian push outward from each eye centre.
    for key in ("eye_l", "eye_r"):
        c = verts[LANDMARKS[key]].mean(axis=0)
        d = v[:n_face, :2] - c[:2]
        w = np.exp(-((d[:, 0] / 0.17) ** 2 + (d[:, 1] / 0.13) ** 2))
        w *= np.clip(v[:n_face, 2] / 0.25, 0.0, 1.0)
        v[:n_face, :2] += d * (0.16 * w)[:, None]

    # 4. Fuller lips, a touch of pout.
    lip_c = verts[LANDMARKS["lips_out"]].mean(axis=0)
    w = _gauss(v[:n_face, 1] - lip_c[1], 0.085) * _gauss(v[:n_face, 0], 0.27)
    w *= np.clip(v[:n_face, 2] / 0.35, 0.0, 1.0)
    v[:n_face, 1] += (v[:n_face, 1] - lip_c[1]) * 0.24 * w
    v[:n_face, 2] += 0.014 * w

    # 5. Smaller, finer nose.
    nose = _gauss(v[:n_face, 1] + 0.20, 0.20) * _gauss(v[:n_face, 0], 0.17)
    nose *= np.clip((v[:n_face, 2] - 0.35) / 0.2, 0.0, 1.0)
    v[:n_face, 2] -= 0.075 * nose
    v[:n_face, 0] *= 1.0 - 0.16 * nose

    # 6. Higher, more arched brows.
    v[:n_face, 1] += 0.030 * base["brow"][:n_face]

    # 7. Fuller, rounder hair volume over the skull (cranium vertices only, and
    #    fading to nothing at the face border so the seam stays closed).
    cr = slice(n_face, n_head)
    zc = v[cr, 2]
    k = np.clip((0.25 - zc) / 0.45, 0.0, 1.0)[:, None]
    scale = np.array([1.09, 1.05, 1.11], dtype=np.float32)
    v[cr] = _CAM_HEAD_C + (v[cr] - _CAM_HEAD_C) * (1.0 + (scale - 1.0) * k)
    return v


def _hair_sheet(n0: int):
    """Long hair: a sheet hung round the back and sides of the head.

    Returns (verts, faces, fade) with face indices already offset by `n0`.
    """
    rows, cols = _HAIR_ROWS, _HAIR_COLS
    verts = np.zeros((rows * cols, 3), dtype=np.float32)
    fade = np.zeros(rows * cols, dtype=np.float32)
    for r in range(rows):
        t = r / (rows - 1)                       # 0 top → 1 bottom
        y = _HAIR_TOP_Y + (_HAIR_BOTTOM_Y - _HAIR_TOP_Y) * t
        rx = 0.74 + 0.22 * t ** 1.1                          # flares over the shoulders
        rz = 0.66 - 0.04 * t
        cz = -0.30 + 0.05 * t
        # reach round to the front less at the top (temples) than at the bottom
        reach = _HAIR_THETA * (0.86 + 0.14 * min(1.0, t * 2.0))
        for c in range(cols):
            th = -reach + 2.0 * reach * c / (cols - 1)
            i = r * cols + c
            # a soft wave so the sheet reads as strands, not a lampshade
            wave = 1.0 + 0.045 * math.sin(c * 2.3 + t * 2.0)
            verts[i] = (math.sin(th) * rx * wave, y, cz - math.cos(th) * rz * wave)
            edge = 1.0 - 0.45 * (abs(th) / reach) ** 4
            fade[i] = max(0.18, (1.0 - 0.72 * t ** 1.6)) * edge
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = n0 + r * cols + c
            b, d = a + 1, a + cols
            e = d + 1
            faces.append((a, b, d))
            faces.append((b, e, d))
    return verts, np.array(faces, dtype=np.int32), fade


def build_girl_mesh() -> dict:
    from core.avatar_mesh import LANDMARKS, _unique_edges, _vertex_normals, get_head_mesh

    base = get_head_mesh()
    n_face, n_head = base["n_face"], base["n_head"]

    v_head = _feminise(base["verts"], base)          # face + cranium + neck
    n_old = v_head.shape[0]

    hv, hf, hfade = _hair_sheet(n_old)
    verts = np.vstack([v_head, hv]).astype(np.float32)
    faces = np.vstack([base["faces"], hf]).astype(np.int32)
    n_all = verts.shape[0]

    # ── normals: outward per part (head: from the skull; neck & hair: radial) ──
    outward = verts - np.array([0.0, verts[:n_head, 1].mean(), 0.0], dtype=np.float32)
    outward[n_head:n_old] = verts[n_head:n_old] - np.array([0.0, 0.0, -0.15], dtype=np.float32)
    outward[n_head:n_old, 1] = 0.0
    outward[n_old:] = verts[n_old:] - np.array([0.0, 0.0, -0.30], dtype=np.float32)
    outward[n_old:, 1] = 0.0
    normals = _vertex_normals(verts, faces, outward)

    # ── per-vertex rigs, extended over the hair (hair follows nothing) ─────────
    def ext(arr, fill):
        return np.concatenate([arr, np.full(n_all - arr.shape[0], fill, dtype=np.float32)])

    jaw = ext(base["jaw"], 0.0)
    brow = ext(base["brow"], 0.0)
    lips = ext(base["lips"], 0.0)
    fade = np.concatenate([base["fade"], hfade]).astype(np.float32)
    lip_c = verts[LANDMARKS["lips_out"]].mean(axis=0)

    # ── which triangles are HAIR (drawn in the hair colour) ───────────────────
    ia, ib, ic = faces[:, 0], faces[:, 1], faces[:, 2]
    touches_cranium = ((ia >= n_face) & (ia < n_head)) | ((ib >= n_face) & (ib < n_head)) \
        | ((ic >= n_face) & (ic < n_head))
    cen = (verts[ia] + verts[ib] + verts[ic]) / 3.0
    on_skull = touches_cranium & ((cen[:, 1] > 0.02) | (cen[:, 2] < -0.50))
    in_sheet = (ia >= n_old) & (ib >= n_old) & (ic >= n_old)
    is_hair = (on_skull | in_sheet) & ~((ia >= n_head) & (ia < n_old))

    # Neck first, then head, then hair sheet last of all (it hangs behind/around).
    group = np.ones(faces.shape[0], dtype=np.float32)
    neck_only = (ia >= n_head) & (ib >= n_head) & (ic >= n_head) & (ia < n_old)
    group[neck_only] = 0.0

    # Wireframe: thin. Keep every third edge on skin, every fourth on hair so the
    # strands read as strands.
    e_all = _unique_edges(faces)
    e_all = e_all[(e_all < n_old).all(axis=1)]          # the sheet is shaded, not wired
    hair_v = np.zeros(n_all, dtype=bool)
    hair_v[np.unique(faces[is_hair & ~in_sheet].ravel())] = True
    is_hair_edge = hair_v[e_all[:, 0]] & hair_v[e_all[:, 1]]
    edges = np.vstack([e_all[~is_hair_edge][::3], e_all[is_hair_edge][::4]])

    return {
        "verts": np.ascontiguousarray(verts, dtype=np.float32),
        "normals": np.ascontiguousarray(normals, dtype=np.float32),
        "faces": np.ascontiguousarray(faces, dtype=np.int32),
        "edges": np.ascontiguousarray(edges, dtype=np.int32),
        "jaw": np.ascontiguousarray(jaw, dtype=np.float32),
        "brow": np.ascontiguousarray(brow, dtype=np.float32),
        "lips": np.ascontiguousarray(lips, dtype=np.float32),
        "lip_centre": np.ascontiguousarray(lip_c, dtype=np.float32),
        "fade": np.ascontiguousarray(fade, dtype=np.float32),
        "face_group": np.ascontiguousarray(group, dtype=np.float32),
        "hair_faces": np.ascontiguousarray(is_hair & ~in_sheet),   # skull cap
        "sheet_faces": np.ascontiguousarray(in_sheet),             # long hair
        "n_body": n_old,
        "landmarks": base["landmarks"],
        "n_face": n_face,
        "n_head": n_head,
        "span": base["span"],          # hair ends above the neck → same layout
    }


def get_girl_mesh() -> dict:
    global _GIRL_CACHE
    if _GIRL_CACHE is None:
        _GIRL_CACHE = build_girl_mesh()
    return _GIRL_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Girl avatar renderer (subclass — the boy HoloAvatar is untouched)
# ─────────────────────────────────────────────────────────────────────────────

def _make_girl_avatar_class():
    """Built lazily so importing this module never needs PyQt at import time."""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF

    from core.avatar import HoloAvatar, _blend, _c, _LUT_N

    class GirlHoloAvatar(HoloAvatar):
        """Same animation and lip-sync as the boy head, on the feminised mesh,
        with hair (own colour), eyelashes and tinted lips."""

        def __init__(self) -> None:
            super().__init__()
            self._load_girl(get_girl_mesh())
            self._hair_luts: dict = {}

        # Same attribute wiring as HoloAvatar.__init__, pointed at the girl mesh.
        def _load_girl(self, mesh: dict) -> None:
            self._v0 = mesh["verts"]
            self._n0 = mesh["normals"]
            self._jaw = mesh["jaw"]
            self._brow_w = mesh["brow"]
            self._lips_w = mesh["lips"]
            self._lip_c = mesh["lip_centre"]
            self._fade = mesh["fade"]
            self._f = mesh["faces"]
            self._fgroup = mesh["face_group"]
            self._fa, self._fb, self._fc = (self._f[:, i] for i in range(3))
            self._e0 = mesh["edges"][:, 0]
            self._e1 = mesh["edges"][:, 1]
            self._lm = mesh["landmarks"]
            lips_in = mesh["landmarks"]["lips_in"]
            self._lip_up = np.concatenate([lips_in[10:], lips_in[:1]])
            self.SPAN = mesh["span"][0] - mesh["span"][1]
            self._v = np.empty((self._v0.shape[0], 3), dtype=np.float32)

            f, g = mesh["faces"], mesh["face_group"]
            cap, sheet = mesh["hair_faces"], mesh["sheet_faces"]
            skin = ~(cap | sheet)
            sel = lambda m: (f[m, 0], f[m, 1], f[m, 2], g[m])
            self._skin, self._cap, self._sheet = sel(skin), sel(cap), sel(sheet)
            self._n_body = mesh["n_body"]

        # Hair gets its own colour ramp, so keep one LUT per (bg, colour).
        def _lut_for(self, bg, col):
            key = (bg.rgb(), col.rgb())
            lut = self._hair_luts.get(key)
            if lut is None:
                if len(self._hair_luts) > 6:
                    self._hair_luts.clear()
                lut = [QBrush(_blend(bg, col, 255.0 * (i + 0.5) / _LUT_N))
                       for i in range(_LUT_N)]
                self._hair_luts[key] = lut
            return lut

        def _lut(self, bg, primary):               # base class calls this
            return self._lut_for(bg, primary)

        def _paint_surface(self, p, xs, ys, norms, verts, primary, bg, amp) -> None:
            saved = (self._fa, self._fb, self._fc, self._fgroup)
            # Hair: a deeper tone of the same theme colour, so it reads as a
            # different material without fighting the HUD palette.
            hair = QColor(primary)
            h, s, v, _a = hair.getHsv()
            hair.setHsv(max(0, h), min(255, int(s * 1.05) + 25), int(v * 0.78))
            try:
                # 0. long hair, BEHIND everything. Two-sided: its normals are
                #    pointed at the camera so the inside of the sheet is lit
                #    and drawn where it shows beside the neck and shoulders.
                two_sided = norms.copy()
                two_sided[self._n_body:] = (0.0, 0.0, 1.0)
                self._fa, self._fb, self._fc, self._fgroup = self._sheet
                HoloAvatar._paint_surface(self, p, xs, ys, two_sided, verts, hair, bg, amp)
                # 1. skin, in the theme colour
                self._fa, self._fb, self._fc, self._fgroup = self._skin
                HoloAvatar._paint_surface(self, p, xs, ys, norms, verts, primary, bg, amp)
                # 2. hair over the skull (hairline sits on the face border)
                self._fa, self._fb, self._fc, self._fgroup = self._cap
                HoloAvatar._paint_surface(self, p, xs, ys, norms, verts, hair, bg, amp)
            finally:
                self._fa, self._fb, self._fc, self._fgroup = saved

        def _paint_features(self, p, xs, ys, norms, r, primary, accent, bg, amp) -> None:
            super()._paint_features(p, xs, ys, norms, r, primary, accent, bg, amp)
            face = max(0.0, math.cos(self._yaw) * math.cos(self._pitch)) ** 2
            if face < 0.02:
                return
            lm = self._lm
            vis = 1.0 - self._blink

            # ── lip colour: the ring between outer and inner lip, tinted ────────
            outer = self._ring(xs, ys, lm["lips_out"])
            inner = self._ring(xs, ys, lm["lips_in"])
            path = QPainterPath()
            path.addPolygon(outer)
            path.closeSubpath()
            hole = QPainterPath()
            hole.addPolygon(inner)
            hole.closeSubpath()
            path = path.subtracted(hole)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_c(accent, 92 * face)))
            p.drawPath(path)

            # ── eyelashes: short flicks off the upper lid, longest at the
            #    outer corner. They follow the blink like the lid does. ────────
            pen = QPen(_c(primary, 235 * face), max(1.0, r * 0.011))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for key, outer_sign in (("eye_l", -1.0), ("eye_r", 1.0)):
                idx = lm[key]
                ex, ey = xs[idx].astype(float), ys[idx].astype(float)
                mid_y = float(ey.mean())
                if vis < 0.999:
                    ey = mid_y + (ey - mid_y) * max(0.04, vis)
                # Ring layout: [0] outer corner, [1..7] lower lid, [8] inner
                # corner, [9..15] upper lid running inner → outer.
                upper = list(range(9, 16))
                n = len(upper)
                k = max(0.0, vis) ** 1.5
                for j, i in enumerate(upper):
                    t = j / (n - 1)                    # 0 inner … 1 outer
                    ln = r * (0.030 + 0.050 * t) * k
                    p.drawLine(QPointF(ex[i], ey[i]),
                               QPointF(ex[i] + outer_sign * ln * (0.25 + 0.75 * t),
                                       ey[i] - ln))
                # winged flick at the outer corner
                ln = r * 0.060 * k
                p.drawLine(QPointF(ex[0], ey[0]),
                           QPointF(ex[0] + outer_sign * ln, ey[0] - ln * 0.55))

    return GirlHoloAvatar


_GIRL_CLS = None


def _girl_class():
    global _GIRL_CLS
    if _GIRL_CLS is None:
        _GIRL_CLS = _make_girl_avatar_class()
    return _GIRL_CLS


class SwitchableAvatar:
    """Drop-in stand-in for HoloAvatar that draws whichever model is active.

    It owns the original (boy) avatar and lazily builds a girl one, and forwards
    every call — step / paint / glance / SPAN … — to the active one, so the HUD
    code that holds `self._avatar` needs no changes at all.
    """

    def __init__(self, boy) -> None:
        object.__setattr__(self, "_boy", boy)
        object.__setattr__(self, "_girl", None)
        object.__setattr__(self, "_girl_failed", False)
        object.__setattr__(self, "_last", None)

    def _current(self):
        want = active()
        if want == GIRL and not self._girl_failed:
            if self._girl is None:
                try:
                    object.__setattr__(self, "_girl", _girl_class()())
                except Exception as e:                    # never break the HUD
                    print(f"[girl_model] could not build girl avatar: {e}")
                    object.__setattr__(self, "_girl_failed", True)
                    return self._boy
            cur = self._girl
        else:
            cur = self._boy
        last = self._last
        if last is not None and last is not cur:
            # carry the clock across so timers (blink, gaze) stay sane
            try:
                cur._t = last._t
                cur._blink_at = cur._t + 1.0
            except Exception:
                pass
        object.__setattr__(self, "_last", cur)
        return cur

    def __getattr__(self, name):                  # step, paint, glance, SPAN, …
        return getattr(self._current(), name)

    def __setattr__(self, name, value):           # e.g. avatar.shaded = False
        setattr(self._boy, name, value)
        if self._girl is not None:
            setattr(self._girl, name, value)


# ─────────────────────────────────────────────────────────────────────────────
# Hooking into the running app
# ─────────────────────────────────────────────────────────────────────────────

_installed = False
_prompt_patched = False


def _patch_prompt_loader() -> None:
    """Append the persona block to the system prompt (main.py's loader)."""
    global _prompt_patched
    if _prompt_patched:
        return
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "_load_system_prompt", None)
        if callable(fn) and not getattr(fn, "_girl_patched", False):
            def wrapped(*a, __orig=fn, **k):
                text = __orig(*a, **k)
                extra = persona_text()
                return f"{text}\n\n{extra}" if extra and isinstance(text, str) else text
            wrapped._girl_patched = True
            mod._load_system_prompt = wrapped
            _prompt_patched = True
            return


def install(ui) -> None:
    """Attach the plug-in to the already-imported `ui` module. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    Hud = getattr(ui, "HudCanvas", None)
    Overlay = getattr(ui, "CustomizeOverlay", None)

    # 1. HUD: wrap the avatar so it can switch model live ─────────────────────
    if Hud is not None:
        orig_init = Hud.__init__

        def hud_init(self, *a, **k):
            orig_init(self, *a, **k)
            try:
                if getattr(self, "_avatar", None) is not None and \
                        not isinstance(self._avatar, SwitchableAvatar):
                    self._avatar = SwitchableAvatar(self._avatar)
            except Exception as e:
                print(f"[girl_model] avatar hook failed: {e}")
            _patch_prompt_loader()          # main.py is fully loaded by now

        Hud.__init__ = hud_init

    # 2. Settings overlay: model row, save, cancel ─────────────────────────────
    if Overlay is not None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton

        Overlay._OH = Overlay._OH + 58          # room for the extra row
        o_init, o_save, o_cancel = Overlay.__init__, Overlay._save, Overlay._cancel

        def _style_btns(self):
            C = ui.C
            for m, b in self._model_btns.items():
                on = (m == self._sel_model)
                b.setChecked(on)
                b.setStyleSheet(
                    f"QPushButton {{ background: {C.PRI if on else '#000d12'}; "
                    f"color: {'#000' if on else C.TEXT}; border: 1px solid "
                    f"{C.PRI if on else C.BORDER}; border-radius: 3px; }}"
                    f"QPushButton:hover {{ border: 1px solid {C.PRI}; }}")

        def _pick(self, model):
            if model == self._sel_model:
                return
            self._sel_model = model
            _style_btns(self)
            # The voice follows the model, unless the current one already fits.
            if not voice_fits(model, self._sel_voice):
                # back to the voice they started with if it suits, else the default
                self._sel_voice = (self._voice_was if voice_fits(model, self._voice_was)
                                   else DEFAULT_VOICE[model])
                self._refresh_voice_btns()
            set_active(model)                     # live preview in the HUD

        def ov_init(self, *a, **k):
            o_init(self, *a, **k)
            try:
                C = ui.C
                self._initial_model = get_model()
                self._voice_was = self._sel_voice
                self._sel_model = self._initial_model
                self._model_btns = {}

                box = QHBoxLayout()
                box.setSpacing(4)
                cap = QLabel("ASSISTANT MODEL")
                cap.setFont(QFont("Courier New", 8))
                cap.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
                lay = self.layout()

                # place it right under the voice row
                where = lay.count()
                first_voice = next(iter(self._voice_btns.values()))
                for i in range(lay.count()):
                    it = lay.itemAt(i)
                    if it.layout() is not None and it.layout().count() and \
                            it.layout().itemAt(0).widget() is first_voice:
                        where = i + 1
                        break

                for m, label in ((BOY, "♂  BOY"), (GIRL, "♀  GIRL")):
                    b = QPushButton(label)
                    b.setCheckable(True)
                    b.setFixedHeight(28)
                    b.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
                    b.setCursor(Qt.CursorShape.PointingHandCursor)
                    b.clicked.connect(lambda _=False, mm=m: _pick(self, mm))
                    self._model_btns[m] = b
                    box.addWidget(b)
                lay.insertSpacing(where, 4)
                lay.insertWidget(where + 1, cap)
                lay.insertLayout(where + 2, box)
                _style_btns(self)
            except Exception as e:
                print(f"[girl_model] settings row failed: {e}")

        def ov_save(self):
            try:
                model = getattr(self, "_sel_model", None)
                if model:
                    changed = model != self._initial_model
                    save_model(model)              # before the signal fires
                    self._initial_model = model
                    o_save(self)
                    if changed:
                        # If the voice did not change the app will not rebuild the
                        # session by itself; do it so the persona text is reloaded.
                        win = self.window()
                        cb = getattr(win, "on_voice_change", None)
                        if callable(cb) and getattr(self, "_voice_was", None) == self._sel_voice:
                            cb()
                    return
            except Exception as e:
                print(f"[girl_model] save hook failed: {e}")
            o_save(self)

        def ov_cancel(self):
            try:
                set_active(getattr(self, "_initial_model", get_model()))
                self._sel_model = self._initial_model
            except Exception:
                pass
            o_cancel(self)

        Overlay.__init__ = ov_init
        Overlay._save = ov_save
        Overlay._cancel = ov_cancel
