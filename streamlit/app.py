"""\
PULSE v2 - Multi-Modal AI Monitor (Streamlit edition, max UI)

Image gestures (Teachable Machine .h5) + sound classes (Teachable Machine .tflite)
+ body pose (MoveNet) on the visitor's browser camera and microphone.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

# Must be set BEFORE TensorFlow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import html as htmllib
import threading
import time
from collections import Counter, deque
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly
from streamlit_webrtc import (AudioProcessorBase, VideoProcessorBase,
                              WebRtcMode, webrtc_streamer)

st.set_page_config(page_title="PULSE | Multi-Modal AI Monitor", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
IMAGE_MODEL_PATH = ROOT / "models" / "image" / "keras_model.h5"
IMAGE_LABELS_PATH = ROOT / "models" / "image" / "labels.txt"
AUDIO_MODEL_PATH = ROOT / "models" / "audio" / "soundclassifier_with_metadata.tflite"
AUDIO_LABELS_PATH = ROOT / "models" / "audio" / "labels.txt"

IMAGE_SIZE = 224
ROI_FRAC = 0.6
TM_AUDIO_RATE = 44100
MOVENET_SIZE = 192
MOVENET_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
POSE_NAMES = ["ARM RAISE", "ARM DOWN", "NAMASTE"]
INFER_LOCK = threading.Lock()

THEMES = {
    "Neon Cyber": ("#22d3ee", "#a78bfa", "#f472b6"),
    "Sunset": ("#fb923c", "#f43f5e", "#facc15"),
    "Matrix": ("#4ade80", "#22d3ee", "#a3e635"),
    "Ice": ("#7dd3fc", "#c4b5fd", "#f0abfc"),
}
KIND_ICON = {"image": "🖐️", "audio": "🎧", "pose": "🕺"}


def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex2bgr(h: str) -> tuple[int, int, int]:
    r, g, b = hex2rgb(h)
    return b, g, r


class TMDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def read_labels(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        labels.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


@st.cache_resource(show_spinner="Loading AI models (first start takes a moment)...")
def load_models() -> dict[str, Any]:
    image_labels = read_labels(IMAGE_LABELS_PATH)
    try:
        image_model = tf.keras.models.load_model(str(IMAGE_MODEL_PATH), compile=False)
    except Exception:
        image_model = tf.keras.models.load_model(
            str(IMAGE_MODEL_PATH), compile=False,
            custom_objects={"DepthwiseConv2D": TMDepthwiseConv2D})
    audio_labels = read_labels(AUDIO_LABELS_PATH)
    probe = tf.lite.Interpreter(model_path=str(AUDIO_MODEL_PATH))
    probe.allocate_tensors()
    window = int(probe.get_input_details()[0]["shape"][1])
    movenet = hub.load(MOVENET_URL).signatures["serving_default"]
    return dict(image_model=image_model, image_labels=image_labels,
                audio_labels=audio_labels, audio_window=window, movenet=movenet)


# ---------------------------------------------------------------------------
# Vision helpers (same logic as the working version)
# ---------------------------------------------------------------------------
def roi_box(h: int, w: int) -> tuple[int, int, int, int]:
    side = int(min(h, w) * ROI_FRAC)
    x1, y1 = (w - side) // 2, (h - side) // 2
    return x1, y1, x1 + side, y1 + side


def predict_image(model: Any, frame: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = roi_box(*frame.shape[:2])
    rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    batch = (rgb.astype(np.float32) / 127.5 - 1.0)[np.newaxis, ...]
    raw = np.clip(np.asarray(model(batch, training=False).numpy()[0], dtype=np.float32), 0, None)
    total = float(raw.sum())
    return raw / total if total > 0 else raw


def run_movenet(movenet: Any, frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = tf.convert_to_tensor(rgb[np.newaxis, ...], dtype=tf.uint8)
    t = tf.cast(tf.image.resize_with_pad(t, MOVENET_SIZE, MOVENET_SIZE), tf.int32)
    return movenet(t)["output_0"].numpy()[0, 0, :, :]


def classify_pose(kp: np.ndarray) -> tuple[str, float]:
    ls, rs, lw, rw = kp[5], kp[6], kp[9], kp[10]
    pts = [ls, rs, lw, rw]
    vis = float(np.mean([p[2] for p in pts]))
    if min(float(p[2]) for p in pts) < 0.20:
        return "NO CLEAR POSE", vis
    sy = (float(ls[0]) + float(rs[0])) / 2.0
    wy = (float(lw[0]) + float(rw[0])) / 2.0
    sw = max(abs(float(ls[1]) - float(rs[1])), 0.08)
    gap = float(np.hypot(float(lw[0]) - float(rw[0]), float(lw[1]) - float(rw[1])))
    if gap / sw < 0.55 and sy - 0.05 <= wy <= sy + 0.48:
        return "NAMASTE", vis
    if float(lw[0]) < float(ls[0]) - 0.12 or float(rw[0]) < float(rs[0]) - 0.12:
        return "ARM RAISE", vis
    return "ARM DOWN", vis


LINKS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]


def draw_skeleton(img: np.ndarray, kp: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    side = max(h, w)
    px, py = (side - w) / 2.0, (side - h) / 2.0
    glow = tuple(int(c * 0.4) for c in color)

    def xy(p: np.ndarray) -> tuple[int, int]:
        return int(float(p[1]) * side - px), int(float(p[0]) * side - py)

    for a, b in LINKS:
        if kp[a, 2] >= 0.25 and kp[b, 2] >= 0.25:
            cv2.line(img, xy(kp[a]), xy(kp[b]), glow, 7, cv2.LINE_AA)
            cv2.line(img, xy(kp[a]), xy(kp[b]), color, 2, cv2.LINE_AA)
    for p in kp:
        if p[2] >= 0.25:
            cv2.circle(img, xy(p), 7, glow, -1, cv2.LINE_AA)
            cv2.circle(img, xy(p), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_roi(img: np.ndarray, col: tuple[int, int, int], spotlight: bool) -> np.ndarray:
    x1, y1, x2, y2 = roi_box(*img.shape[:2])
    if spotlight:                                  # dim everything outside the hand zone
        out = (img * 0.5).astype(np.uint8)
        out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        img = out
    ln = 26
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * ln, cy), col, 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * ln), col, 3, cv2.LINE_AA)
    sy = int(y1 + (np.sin(time.time() * 2.2) * 0.5 + 0.5) * (y2 - y1))      # scanning line
    overlay = img.copy()
    cv2.line(overlay, (x1 + 4, sy), (x2 - 4, sy), col, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, "HAND ZONE", (x1 + 6, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


def hud_chip(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int],
             right: bool = False) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    if right:
        x = x - tw - 16
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 8, y - th - 8), (x + tw + 8, y + 8), (18, 12, 8), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# WebRTC processors
# ---------------------------------------------------------------------------
class VisionProcessor(VideoProcessorBase):
    def __init__(self, models: dict[str, Any]) -> None:
        self.m = models
        self.lock = threading.Lock()
        self.skip, self.mirror, self.show_skeleton, self.show_roi = 2, True, True, True
        self.spotlight, self.img_thr = True, 0.5
        self.c_img, self.c_pose = (238, 211, 34), (182, 114, 244)
        self.n = 0
        self.kp: np.ndarray | None = None
        self.img_hist: deque[np.ndarray] = deque(maxlen=4)
        self.pose_hist: deque[str] = deque(maxlen=5)
        self.fps_hist: deque[float] = deque(maxlen=15)
        self.t_prev = time.monotonic()
        self.result: dict[str, Any] = dict(probs=None, pose="NO CLEAR POSE", vis=0.0, fps=0.0, error=None)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        if self.mirror:
            img = cv2.flip(img, 1)

        self.n += 1
        if self.kp is None or self.n % max(1, int(self.skip)) == 0:
            try:
                with INFER_LOCK:                       # classify the CLEAN frame first
                    probs = predict_image(self.m["image_model"], img)
                    kp = run_movenet(self.m["movenet"], img)
                self.img_hist.append(probs)
                self.kp = kp
                pose, vis = classify_pose(kp)
                self.pose_hist.append(pose)
                with self.lock:
                    self.result.update(probs=np.mean(self.img_hist, axis=0), vis=vis, error=None,
                                       pose=Counter(self.pose_hist).most_common(1)[0][0])
            except Exception as exc:
                with self.lock:
                    self.result["error"] = str(exc)[:80]

        now = time.monotonic()
        self.fps_hist.append(1.0 / max(now - self.t_prev, 1e-3))
        self.t_prev = now
        with self.lock:
            self.result["fps"] = fps = float(np.mean(self.fps_hist))
            probs, pose = self.result["probs"], self.result["pose"]

        if self.show_roi:
            img = draw_roi(img, self.c_img, self.spotlight)
        if self.show_skeleton and self.kp is not None:
            draw_skeleton(img, self.kp, self.c_pose)

        hud_chip(img, f"LIVE  {fps:.0f} FPS", 16, 30, (140, 230, 120))
        if probs is not None:
            idx = int(np.argmax(probs))
            labels = self.m["image_labels"]
            conf = float(probs[idx])
            text = f"{labels[idx]}  {conf:.0%}" if conf >= self.img_thr else "..."
            hud_chip(img, text, img.shape[1] - 16, 30, self.c_img, right=True)
        hud_chip(img, f"POSE  {pose}", 16, img.shape[0] - 18, self.c_pose)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


class SoundProcessor(AudioProcessorBase):
    def __init__(self, models: dict[str, Any]) -> None:
        self.window = int(models["audio_window"])
        self.interp = tf.lite.Interpreter(model_path=str(AUDIO_MODEL_PATH))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.buf = np.zeros(0, dtype=np.float32)
        self.last_run = 0.0
        self.lock = threading.Lock()
        self.result: dict[str, Any] = dict(scores=None, rms=0.0, updated=None, error=None)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()
        ch = max(1, len(frame.layout.channels))
        mono = arr.astype(np.float32).mean(axis=0) if frame.format.is_planar \
            else arr.astype(np.float32).reshape(-1, ch).mean(axis=1)
        if np.issubdtype(arr.dtype, np.integer):
            mono /= float(np.iinfo(arr.dtype).max)
        rate = int(frame.sample_rate)
        need = int(np.ceil(self.window * rate / TM_AUDIO_RATE)) + 2
        self.buf = np.concatenate([self.buf, mono])[-need * 2:]

        if len(self.buf) >= need and time.monotonic() - self.last_run >= 0.5:
            self.last_run = time.monotonic()
            try:
                x = np.clip(np.nan_to_num(self.buf[-need:]), -1.0, 1.0)
                if rate != TM_AUDIO_RATE:
                    g = gcd(rate, TM_AUDIO_RATE)
                    x = resample_poly(x, TM_AUDIO_RATE // g, rate // g).astype(np.float32)
                x = x[:self.window]
                if x.size < self.window:
                    x = np.pad(x, (0, self.window - x.size))
                self.interp.set_tensor(self.inp["index"], x.reshape(self.inp["shape"]).astype(np.float32))
                self.interp.invoke()
                scores = np.array(self.interp.get_tensor(self.out["index"])[0])
                with self.lock:
                    self.result.update(scores=scores, rms=float(np.sqrt(np.mean(x ** 2))),
                                       updated=time.time(), error=None)
            except Exception as exc:
                with self.lock:
                    self.result["error"] = str(exc)[:80]

        silent = av.AudioFrame.from_ndarray(np.zeros_like(arr), format=frame.format.name,
                                            layout=frame.layout.name)
        silent.sample_rate = frame.sample_rate
        return silent


# ---------------------------------------------------------------------------
# Page setup, models, sidebar
# ---------------------------------------------------------------------------
missing = [p.relative_to(ROOT) for p in (IMAGE_MODEL_PATH, IMAGE_LABELS_PATH, AUDIO_MODEL_PATH, AUDIO_LABELS_PATH)
           if not p.exists()]
if missing:
    st.error("Missing model files: " + ", ".join(str(m) for m in missing) +
             ". Copy your `models/` folder next to `app.py`.")
    st.stop()

models = load_models()
st.session_state.setdefault("t0", time.time())

with st.sidebar:
    st.markdown("### 🫀 PULSE · Control room")
    theme = st.selectbox("🎨 Theme", list(THEMES))
    img_thr = st.slider("Gesture confidence", 0.30, 0.95, 0.50, 0.05)
    aud_thr = st.slider("Sound confidence", 0.30, 0.95, 0.60, 0.05)
    speed = st.slider("Run AI every N frames", 1, 5, 2,
                      help="Higher = smoother video on slow servers, slightly laggier predictions.")
    st.markdown("**Camera view**")
    mirror = st.toggle("Mirror camera", True, help="Teachable Machine models are trained on mirrored views.")
    show_roi = st.toggle("Show hand zone", True)
    spot = st.toggle("Spotlight hand zone", True, help="Dims everything outside the hand zone.")
    show_skel = st.toggle("Show skeleton", True)
    toasts = st.toggle("Pop-up toasts on new events", False)
    if st.button("♻️ Reset session", use_container_width=True):
        for k in ("events", "counts", "last"):
            st.session_state.pop(k, None)
        st.session_state["t0"] = time.time()
    st.divider()
    st.caption("Gestures: " + ", ".join(models["image_labels"]))
    st.caption("Sounds: " + ", ".join(models["audio_labels"]))

C_IMAGE, C_AUDIO, C_POSE = THEMES[theme]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500&display=swap');
:root { --c-image:__C1__; --c-audio:__C2__; --c-pose:__C3__;
        --c1:__R1__; --c2:__R2__; --c3:__R3__; }
html, body, .stApp { font-family:'Space Grotesk',sans-serif; }
.stApp { background:
  linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px) 0 0/46px 46px,
  linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px) 0 0/46px 46px,
  radial-gradient(900px 520px at 6% -8%, rgba(var(--c1),.20), transparent 62%),
  radial-gradient(800px 520px at 100% 0%, rgba(var(--c3),.17), transparent 62%),
  radial-gradient(900px 600px at 50% 120%, rgba(var(--c2),.18), transparent 62%), #060914; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.4rem; max-width:1360px; }
section[data-testid="stSidebar"] { background:rgba(8,12,26,.88); border-right:1px solid rgba(255,255,255,.07); }
iframe[title*="webrtc"], video { border-radius:22px; }
iframe[title*="webrtc"] { box-shadow:0 0 0 1px rgba(var(--c1),.45), 0 24px 70px rgba(var(--c1),.15); background:#050810; }
.brand { font-size:3.6rem; font-weight:700; letter-spacing:.06em; line-height:1;
  background:linear-gradient(90deg,var(--c-image),var(--c-audio),var(--c-pose),var(--c-image)); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:flow 9s linear infinite; }
@keyframes flow { to { background-position:300% 0; } }
.tag { color:#94a3b8; font-size:1.05rem; margin-top:8px; }
.live-wrap { display:flex; justify-content:flex-end; padding-top:10px; }
.live { display:inline-flex; align-items:center; gap:9px; padding:8px 16px; border-radius:999px;
  background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12); font-family:'JetBrains Mono',monospace; font-size:.84rem; color:#e2e8f0; }
.dot { width:10px; height:10px; border-radius:50%; background:#475569; }
.dot.on { background:#34d399; box-shadow:0 0 12px #34d399; }
.stats { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:10px 0 18px; }
.stat { padding:12px 16px; border-radius:16px; background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08); }
.stat small { display:block; color:#64748b; letter-spacing:.14em; text-transform:uppercase; font-size:.64rem; margin-bottom:4px; }
.stat b { color:#f1f5f9; font-size:1.12rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block; }
.card { position:relative; padding:16px 18px 14px; margin-bottom:14px; border-radius:22px; overflow:hidden;
  background:linear-gradient(160deg,rgba(255,255,255,.075),rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.1); box-shadow:0 14px 44px rgba(0,0,0,.4); }
.card::before { content:''; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent); box-shadow:0 0 20px var(--accent); }
.card::after { content:''; position:absolute; right:-60px; top:-60px; width:180px; height:180px; border-radius:50%;
  background:radial-gradient(circle,var(--accent),transparent 70%); opacity:.16; }
.card-top { display:flex; justify-content:space-between; align-items:center; }
.chip { font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); font-weight:700; }
.pct { font-family:'JetBrains Mono',monospace; color:#e2e8f0; font-size:.95rem; }
.card-main { display:flex; align-items:center; gap:16px; margin-top:10px; }
.ring { position:relative; width:84px; height:84px; flex:0 0 84px; }
.ring svg { width:84px; height:84px; transform:rotate(-90deg); }
.trk { fill:none; stroke:rgba(255,255,255,.09); stroke-width:7; }
.arc { fill:none; stroke:var(--accent); stroke-width:7; stroke-linecap:round; filter:drop-shadow(0 0 6px var(--accent)); }
.ring span { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:2.1rem; }
.value { font-size:1.75rem; font-weight:700; color:#f8fafc; line-height:1.15; }
.sub { color:#94a3b8; font-size:.82rem; margin-top:3px; }
.rows { margin-top:12px; }
.row { display:grid; grid-template-columns:96px 1fr 40px; gap:8px; align-items:center; font-size:.78rem; color:#cbd5e1; margin:5px 0; }
.row span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.row b { font-family:'JetBrains Mono',monospace; font-weight:500; text-align:right; color:#94a3b8; }
.mini { height:5px; background:rgba(255,255,255,.07); border-radius:99px; overflow:hidden; }
.mini i { display:block; height:100%; background:var(--accent); opacity:.9; }
.eq { display:flex; align-items:flex-end; gap:3px; height:58px; margin-top:12px; }
.eq i { flex:1; border-radius:3px; background:linear-gradient(180deg,#fff,var(--accent)); box-shadow:0 0 8px var(--accent); }
.pose-wrap { display:flex; align-items:center; gap:18px; margin-top:6px; }
.pills { display:flex; gap:8px; flex-wrap:wrap; }
.pill { padding:6px 13px; border-radius:999px; font-size:.76rem; border:1px solid rgba(255,255,255,.14); color:#94a3b8; }
.pill.on { color:#0b1020; background:var(--accent); border-color:var(--accent); font-weight:700; box-shadow:0 0 18px var(--accent); }
.scene { padding:16px 18px; border-radius:22px; margin-top:16px;
  background:linear-gradient(90deg,rgba(var(--c1),.14),rgba(var(--c2),.14),rgba(var(--c3),.14)); border:1px solid rgba(255,255,255,.1); }
.scene > small { display:block; color:#94a3b8; letter-spacing:.18em; text-transform:uppercase; font-size:.66rem; margin-bottom:10px; }
.fusion { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.f { text-align:center; padding:12px 8px; border-radius:16px; background:rgba(0,0,0,.28); border:1px solid rgba(255,255,255,.08);
  border-bottom:3px solid var(--accent); }
.f .big { display:block; font-size:2.4rem; }
.f b { display:block; color:#f8fafc; font-size:1rem; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.f i { color:var(--accent); font-style:normal; font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; }
.tl { margin-top:16px; padding-left:6px; border-left:2px solid rgba(255,255,255,.08); }
.tl-item { position:relative; padding:5px 0 5px 18px; font-size:.88rem; color:#e2e8f0; }
.tl-item b { font-family:'JetBrains Mono',monospace; font-weight:500; color:#64748b; margin-right:8px; font-size:.78rem; }
.tl-dot { position:absolute; left:-7px; top:11px; width:11px; height:11px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); }
.tl-ico { margin-right:6px; }
.tl-title { color:#94a3b8; letter-spacing:.18em; text-transform:uppercase; font-size:.66rem; margin-top:18px; }
.guide { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.g { padding:14px; border-radius:16px; background:rgba(0,0,0,.28); border:1px solid rgba(255,255,255,.08); color:#cbd5e1; font-size:.9rem; }
.g b { display:block; color:#f8fafc; font-size:1.5rem; margin-bottom:4px; }
.step { padding:18px; border-radius:18px; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); height:100%; }
.step h4 { margin:0 0 6px; color:#f8fafc; } .step p { color:#94a3b8; font-size:.9rem; margin:0; }
.foot { text-align:center; color:#475569; font-size:.8rem; margin-top:28px; }
@media (max-width: 900px) { .stats { grid-template-columns:repeat(2,1fr); } .fusion, .guide { grid-template-columns:1fr; } }
</style>
"""
for token, val in (("__C1__", C_IMAGE), ("__C2__", C_AUDIO), ("__C3__", C_POSE)):
    CSS = CSS.replace(token, val)
for token, val in (("__R1__", C_IMAGE), ("__R2__", C_AUDIO), ("__R3__", C_POSE)):
    CSS = CSS.replace(token, "{},{},{}".format(*hex2rgb(val)))
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def flat(s: str) -> str:
    """Remove indentation/newlines so Markdown never turns the HTML into a code block."""
    return "".join(line.strip() for line in s.splitlines())


def esc(s: Any) -> str:
    return htmllib.escape(str(s))


def norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def emoji_for(label: str, kind: str) -> str:
    k = norm(label)
    tables = {
        "image": {"thumbsup": "👍", "thumbsdown": "👎", "palm": "✋", "open": "✋", "fist": "✊",
                  "peace": "✌️", "victory": "✌️", "point": "☝️", "ok": "👌", "wave": "👋",
                  "rock": "🤘", "none": "🫥", "nothing": "🫥", "background": "🫥"},
        "audio": {"clap": "👏", "knock": "🚪", "whistle": "🎶", "snap": "🫰", "background": "🔇",
                  "noise": "🔇", "speech": "🗣️", "voice": "🗣️", "music": "🎵", "cough": "🤧",
                  "laugh": "😂", "listening": "👂"},
        "pose": {"armraise": "🙋", "armdown": "🧍", "namaste": "🙏", "noclear": "🫥"},
    }
    for key, e in tables[kind].items():
        if key in k:
            return e
    return {"image": "🖐️", "audio": "🔊", "pose": "🕺"}[kind]


def bars(items: list[tuple[str, float]]) -> str:
    return '<div class="rows">' + "".join(
        f'<div class="row"><span>{esc(n)}</span><div class="mini"><i style="width:{p * 100:.0f}%"></i></div>'
        f'<b>{p * 100:.0f}%</b></div>' for n, p in items) + "</div>"


def ring(conf: float, emoji: str) -> str:
    c = 2 * 3.14159 * 34
    d = c * max(0.0, min(1.0, conf))
    return (f'<div class="ring"><svg viewBox="0 0 84 84"><circle class="trk" cx="42" cy="42" r="34"></circle>'
            f'<circle class="arc" cx="42" cy="42" r="34" style="stroke-dasharray:{d:.1f} {c:.1f}"></circle></svg>'
            f'<span>{emoji}</span></div>')


def card(title: str, value: str, sub: str, conf: float, accent: str, emoji: str,
         extra: str = "", chip_icon: str = "") -> str:
    pct = int(max(0.0, min(1.0, conf)) * 100)
    return flat(f"""
    <div class="card" style="--accent:{accent}">
      <div class="card-top"><span class="chip">{chip_icon} {esc(title)}</span><span class="pct">{pct}%</span></div>
      <div class="card-main">{ring(conf, emoji)}<div><div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div></div>
      {extra}
    </div>""")


def equalizer(level: float, active: bool) -> str:
    rng = np.random.default_rng(int(time.time() * 6))
    n = 30
    env = np.sin(np.linspace(0.25, np.pi - 0.25, n))
    amp = min(1.0, level * 10) if active else 0.03
    hs = 5 + env * (rng.random(n) * 0.6 + 0.4) * amp * 50
    return '<div class="eq">' + "".join(f'<i style="height:{h:.0f}px"></i>' for h in hs) + "</div>"


def pose_figure(label: str, accent: str) -> str:
    arms = {
        "ARM DOWN": "M44 46 L36 68 L34 90 M76 46 L84 68 L86 90",
        "ARM RAISE": "M44 46 L36 68 L34 90 M76 46 L90 30 L94 8",
        "NAMASTE": "M44 46 L46 66 L60 56 M76 46 L74 66 L60 56",
    }
    known = label in arms
    path = arms.get(label, arms["ARM DOWN"])
    return (f'<svg viewBox="0 0 120 140" width="92" height="108" style="opacity:{1 if known else .3}">'
            f'<defs><filter id="glow"><feGaussianBlur stdDeviation="2.6" result="b"></feGaussianBlur>'
            f'<feMerge><feMergeNode in="b"></feMergeNode><feMergeNode in="SourceGraphic"></feMergeNode></feMerge></filter></defs>'
            f'<g stroke="{accent}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" fill="none" filter="url(#glow)">'
            f'<circle cx="60" cy="22" r="12"></circle>'
            f'<path d="M60 34 L60 88 M60 88 L46 128 M60 88 L74 128 M44 46 L76 46"></path>'
            f'<path d="{path}"></path></g></svg>')


def log_event(kind: str, text: str) -> None:
    ev = st.session_state.setdefault("events", [])
    ev.insert(0, (datetime.now().strftime("%H:%M:%S"), kind, text))
    del ev[7:]
    st.session_state.setdefault("counts", {"image": Counter(), "audio": Counter(), "pose": Counter()})
    st.session_state["counts"][kind][text] += 1
    if toasts:
        st.toast(f"{emoji_for(text, kind)} {text}")


def top_of(kind: str) -> str:
    c = st.session_state.get("counts", {}).get(kind)
    if not c:
        return "-"
    name = c.most_common(1)[0][0]
    return f"{emoji_for(name, kind)} {name}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
tab_live, tab_how, tab_deploy = st.tabs(["🎛️ Live monitor", "🧠 How it works", "🚀 Deploy notes"])

with tab_live:
    h1, h2 = st.columns([3, 1])
    h1.markdown(flat("""<div class="brand">PULSE</div>
        <div class="tag">See your gestures, hear your sounds, read your body — one live dashboard.</div>"""),
                unsafe_allow_html=True)
    ph_live = h2.empty()
    ph_stats = st.empty()
    left, right = st.columns([3, 2], gap="large")
    with left:
        def rtc_config() -> dict[str, Any]:
            servers: list[dict[str, Any]] = [{"urls": ["stun:stun.l.google.com:19302",
                                                       "stun:stun1.l.google.com:19302"]}]
            try:  # optional TURN server in secrets, needed on many cloud networks
                t = st.secrets["turn"]
                servers.append({"urls": list(t["urls"]), "username": t["username"],
                                "credential": t["credential"]})
            except Exception:
                pass
            return {"iceServers": servers}

        ctx = webrtc_streamer(
            key="pulse", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config(),
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                                      "audio": True},
            video_processor_factory=lambda: VisionProcessor(models),
            audio_processor_factory=lambda: SoundProcessor(models),
            async_processing=True,
            video_html_attrs={"style": {"width": "100%"}, "controls": False, "autoPlay": True, "muted": True},
        )
        ph_scene = st.empty()
        ph_log = st.empty()
    with right:
        ph_image, ph_audio, ph_pose = st.empty(), st.empty(), st.empty()

with tab_how:
    c1, c2, c3 = st.columns(3)
    c1.markdown(flat("""<div class="step"><h4>🖐️ Gesture vision</h4><p>Your Teachable Machine MobileNet
    model looks at a 60% centre crop of the camera frame (the hand zone), scaled to 224x224 and
    normalised to [-1, 1], then averaged over a few frames to stop flicker.</p></div>"""), unsafe_allow_html=True)
    c2.markdown(flat("""<div class="step"><h4>🎧 Sound intelligence</h4><p>The browser streams microphone
    audio. It is resampled to 44.1 kHz and fed in 44,032-sample windows to your TFLite sound classifier
    about twice per second.</p></div>"""), unsafe_allow_html=True)
    c3.markdown(flat("""<div class="step"><h4>🕺 Body pose</h4><p>MoveNet Lightning finds 17 body keypoints.
    Simple geometric rules turn shoulders and wrists into ARM RAISE, ARM DOWN or NAMASTE, smoothed over
    five frames.</p></div>"""), unsafe_allow_html=True)
    st.markdown("**Pipeline:** Browser camera + mic → WebRTC → Python processors (threads) → live dashboard.")

with tab_deploy:
    st.markdown("""
- Put `models/` next to `app.py`, then push the folder to GitHub (keep files under 100 MB).
- On Streamlit Community Cloud choose **Python 3.11** under *Advanced settings*.
- If the video never starts on the deployed site, add a TURN server in *Secrets*:
  ```toml
  [turn]
  urls = ["turn:YOUR_TURN_HOST:3478"]
  username = "user"
  credential = "pass"
  ```
- TensorFlow is heavy. If the app runs out of memory, deploy on a machine with more RAM
  (for example Hugging Face Spaces or a small VPS).
""")

st.markdown('<div class="foot">Built with Teachable Machine · MoveNet · Streamlit · WebRTC</div>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live rendering
# ---------------------------------------------------------------------------
def render(v: dict[str, Any] | None, a: dict[str, Any] | None, playing: bool) -> None:
    # --- live badge
    fps_text = f"{v['fps']:.0f} FPS" if v else "standby"
    state_text = "LIVE" if playing else "IDLE"
    dot_cls = "on" if playing else ""
    ph_live.markdown(flat(f"""<div class="live-wrap"><div class="live"><span class="dot {dot_cls}"></span>
        {state_text} · {fps_text}</div></div>"""), unsafe_allow_html=True)

    # --- gesture card
    i_label, i_conf, i_sub, i_rows = "Waiting for camera", 0.0, "Press START and allow the camera", ""
    if v and v.get("probs") is not None:
        p = v["probs"]
        idx = int(np.argmax(p))
        i_conf = float(p[idx])
        i_label = models["image_labels"][idx] if i_conf >= img_thr else "Uncertain"
        i_sub = v["error"] or ("Hold your hand inside the hand zone" if i_conf < img_thr else "Gesture locked in")
        i_rows = bars(sorted(zip(models["image_labels"], p), key=lambda t: -t[1])[:5])
    ph_image.markdown(card("Gesture vision", i_label, i_sub, i_conf, C_IMAGE, emoji_for(i_label, "image"),
                           i_rows, "🖐️"), unsafe_allow_html=True)

    # --- sound card
    a_label, a_conf, a_sub, a_rows, level = "Waiting for microphone", 0.0, "Allow microphone access", "", 0.0
    fresh = bool(a and a.get("scores") is not None and a["updated"] and time.time() - a["updated"] < 4)
    if fresh:
        s = a["scores"]
        idx = int(np.argmax(s))
        a_conf = float(s[idx])
        name = models["audio_labels"][idx] if idx < len(models["audio_labels"]) else str(idx)
        a_label = name if a_conf >= aud_thr else "Listening..."
        level = float(a["rms"])
        a_sub = f"input level {level:.3f}"
        a_rows = bars(sorted(zip(models["audio_labels"], s), key=lambda t: -t[1])[:4])
    elif a and a.get("error"):
        a_sub = a["error"]
    ph_audio.markdown(card("Sound intelligence", a_label, a_sub, a_conf, C_AUDIO, emoji_for(a_label, "audio"),
                           equalizer(level, fresh) + a_rows, "🎧"), unsafe_allow_html=True)

    # --- pose card
    live_pose = bool(v and v.get("probs") is not None)
    p_label, p_vis = (v["pose"], v["vis"]) if live_pose else ("Waiting for camera", 0.0)
    pills = '<div class="pills">' + "".join(
        f'<span class="pill {"on" if n == p_label else ""}">{n.title()}</span>' for n in POSE_NAMES) + "</div>"
    extra = f'<div class="pose-wrap">{pose_figure(p_label, C_POSE)}{pills}</div>'
    ph_pose.markdown(card("Body pose · MoveNet", p_label.title(), f"landmark visibility {p_vis:.0%}", p_vis,
                          C_POSE, emoji_for(p_label, "pose"), extra, "🕺"), unsafe_allow_html=True)

    # --- events
    if playing and live_pose:
        last = st.session_state.setdefault("last", {})
        for kind, label, conf, thr in (("image", i_label, i_conf, img_thr),
                                       ("audio", a_label, a_conf, aud_thr),
                                       ("pose", p_label, p_vis, 0.0)):
            ok = conf >= thr and label not in ("Uncertain", "Listening...", "NO CLEAR POSE") \
                and "background" not in label.lower()
            if ok and last.get(kind) != label:
                log_event(kind, label)
            if ok:
                last[kind] = label

    # --- stats row
    secs = int(time.time() - st.session_state["t0"])
    total = sum(sum(c.values()) for c in st.session_state.get("counts", {}).values())
    tiles = [("Session", f"{secs // 60:02d}:{secs % 60:02d}"), ("Events", str(total)),
             ("Top gesture", top_of("image")), ("Top sound", top_of("audio")), ("Top pose", top_of("pose"))]
    ph_stats.markdown('<div class="stats">' + "".join(
        f'<div class="stat"><small>{k}</small><b>{esc(val)}</b></div>' for k, val in tiles) + "</div>",
        unsafe_allow_html=True)

    # --- fusion view / onboarding
    if playing and live_pose:
        ph_scene.markdown(flat(f"""<div class="scene"><small>Fusion view · everything at once</small><div class="fusion">
            <div class="f" style="--accent:{C_IMAGE}"><span class="big">{emoji_for(i_label, 'image')}</span><b>{esc(i_label)}</b><i>Gesture</i></div>
            <div class="f" style="--accent:{C_AUDIO}"><span class="big">{emoji_for(a_label, 'audio')}</span><b>{esc(a_label)}</b><i>Sound</i></div>
            <div class="f" style="--accent:{C_POSE}"><span class="big">{emoji_for(p_label, 'pose')}</span><b>{esc(p_label.title())}</b><i>Pose</i></div>
            </div></div>"""), unsafe_allow_html=True)
    else:
        ph_scene.markdown(flat("""<div class="scene"><small>Get started</small><div class="guide">
            <div class="g"><b>1️⃣</b>Press <b style="display:inline;font-size:1rem">START</b> above</div>
            <div class="g"><b>2️⃣</b>Allow camera + microphone</div>
            <div class="g"><b>3️⃣</b>Show a gesture, make a sound, strike a pose</div></div></div>"""),
                          unsafe_allow_html=True)

    events = st.session_state.get("events", [])
    if events:
        items = "".join(
            f'<div class="tl-item" style="--accent:var(--c-{k})"><span class="tl-dot"></span><b>{t}</b>'
            f'<span class="tl-ico">{emoji_for(x, k)}</span>{esc(x)}</div>' for t, k, x in events)
        ph_log.markdown(f'<div class="tl-title">Activity timeline</div><div class="tl">{items}</div>',
                        unsafe_allow_html=True)
    else:
        ph_log.markdown('<div class="tl-title">Activity timeline</div><div class="tl">'
                        '<div class="tl-item" style="--accent:#64748b"><span class="tl-dot"></span>'
                        'Events will appear here as things happen ✨</div></div>', unsafe_allow_html=True)


if ctx.state.playing:
    while ctx.state.playing:
        vp, ap = ctx.video_processor, ctx.audio_processor
        v = a = None
        if vp is not None:
            vp.skip, vp.mirror, vp.show_roi, vp.show_skeleton = speed, mirror, show_roi, show_skel
            vp.spotlight, vp.img_thr = spot, img_thr
            vp.c_img, vp.c_pose = hex2bgr(C_IMAGE), hex2bgr(C_POSE)
            with vp.lock:
                v = dict(vp.result)
        if ap is not None:
            with ap.lock:
                a = dict(ap.result)
        render(v, a, True)
        time.sleep(0.2)
else:
    render(None, None, False)