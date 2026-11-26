
from __future__ import annotations

import os


os.environ["TF_USE_LEGACY_KERAS"] = "1"      
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import threading
import time
from collections import Counter, deque
from math import gcd
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

try:
    import tf_keras  
except ImportError:
    if tf.__version__ >= "2.16":
        raise SystemExit("Missing package. Run:  pip install tf_keras")

try: 
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

ROOT = Path(__file__).resolve().parent
IMAGE_MODEL_PATH = ROOT / "models" / "image" / "keras_model.h5"
IMAGE_LABELS_PATH = ROOT / "models" / "image" / "labels.txt"
AUDIO_MODEL_PATH = ROOT / "models" / "audio" / "soundclassifier_with_metadata.tflite"
AUDIO_LABELS_PATH = ROOT / "models" / "audio" / "labels.txt"
OUTPUT_DIR = ROOT / "output"
VIDEO_PATH = OUTPUT_DIR / "b4_combined.mp4"
SCREENSHOT_PATH = OUTPUT_DIR / "b4_combined_screenshot.jpg"

CAMERA_INDEX = 0
IMAGE_SIZE = 224
ROI_FRAC = 0.6                 
IMAGE_MIN_CONF = 0.50         
IMAGE_SMOOTHING_FRAMES = 4
TM_AUDIO_RATE = 44100          
MOVENET_SIZE = 192
MOVENET_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
POSE_SMOOTHING_FRAMES = 5
WINDOW = "B4 | Multi-Modal Smart Monitor"
DASH_W, DASH_H = 1280, 720
RECORD_FPS = 20.0

BG = (20, 25, 36)
PANEL = (31, 40, 55)
TEXT = (235, 241, 248)
MUTED = (160, 176, 194)
CYAN = (255, 190, 70)
GREEN = (95, 220, 140)
ORANGE = (40, 170, 255)
RED = (80, 90, 245)


class TMDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Fallback shim for a `groups` field in some Teachable Machine H5 files."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def read_labels(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        labels.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line)
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def load_models():
    required = [IMAGE_MODEL_PATH, IMAGE_LABELS_PATH, AUDIO_MODEL_PATH, AUDIO_LABELS_PATH]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n - " + "\n - ".join(missing))

    print("[1/3] Loading image model...")
    image_labels = read_labels(IMAGE_LABELS_PATH)
    try:  
        image_model = tf.keras.models.load_model(str(IMAGE_MODEL_PATH), compile=False)
    except Exception as exc:
        print(f"     Plain load failed ({exc}); retrying with compatibility shim...")
        image_model = tf.keras.models.load_model(
            str(IMAGE_MODEL_PATH), compile=False,
            custom_objects={"DepthwiseConv2D": TMDepthwiseConv2D},
        )
    n_out = int(image_model.output_shape[-1])
    if len(image_labels) != n_out:
        raise ValueError(f"Image labels={len(image_labels)} but model outputs={n_out}.")
    print(f"     Image model ready: {len(image_labels)} classes | input={image_model.input_shape}")

    print("[2/3] Loading audio model (raw TFLite interpreter)...")
    audio_labels = read_labels(AUDIO_LABELS_PATH)
    audio_model = Interpreter(model_path=str(AUDIO_MODEL_PATH))
    audio_model.allocate_tensors()
    window = int(audio_model.get_input_details()[0]["shape"][1])
    n_audio_out = int(audio_model.get_output_details()[0]["shape"][-1])
    if len(audio_labels) != n_audio_out:
        print(f"     [WARN] Audio labels={len(audio_labels)} but model outputs={n_audio_out}.")
    print(f"     Audio model ready: {len(audio_labels)} classes | window={window} samples")

    print("[3/3] Loading MoveNet pose detector (first run may download the model)...")
    movenet_module = hub.load(MOVENET_URL)
    movenet = movenet_module.signatures["serving_default"]
    print("     MoveNet ready: 17 keypoints")
    return image_model, image_labels, audio_model, audio_labels, movenet



def centre_square_crop(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    side = int(min(h, w) * ROI_FRAC)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2, y2 = x1 + side, y1 + side
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_image(frame: np.ndarray) -> np.ndarray:
    square, _ = centre_square_crop(frame)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    return normalized[np.newaxis, ...]


def predict_image(model: tf.keras.Model, frame: np.ndarray) -> np.ndarray:
    batch = preprocess_image(frame)
    raw = np.asarray(model(batch, training=False).numpy()[0], dtype=np.float32)
    raw = np.clip(raw, 0.0, None)
    total = float(raw.sum())
    if total > 0:
        raw /= total
    return raw



def run_movenet(movenet: Any, frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(rgb[np.newaxis, ...], dtype=tf.uint8)
    tensor = tf.image.resize_with_pad(tensor, MOVENET_SIZE, MOVENET_SIZE)
    tensor = tf.cast(tensor, tf.int32)
    output = movenet(tensor)
    return output["output_0"].numpy()[0, 0, :, :]  


def classify_pose(kp: np.ndarray) -> tuple[str, float]:
    ls, rs, lw, rw = kp[5], kp[6], kp[9], kp[10]
    points = [ls, rs, lw, rw]
    visibility = float(np.mean([p[2] for p in points]))
    if min(float(p[2]) for p in points) < 0.20:
        return "NO CLEAR POSE", visibility

    shoulder_y = (float(ls[0]) + float(rs[0])) / 2.0
    wrist_y = (float(lw[0]) + float(rw[0])) / 2.0
    shoulder_width = max(abs(float(ls[1]) - float(rs[1])), 0.08)
    wrist_gap = float(np.hypot(float(lw[0]) - float(rw[0]), float(lw[1]) - float(rw[1])))

  
    if wrist_gap / shoulder_width < 0.55 and shoulder_y - 0.05 <= wrist_y <= shoulder_y + 0.48:
        return "NAMASTE", visibility
    
    if float(lw[0]) < float(ls[0]) - 0.12 or float(rw[0]) < float(rs[0]) - 0.12:
        return "ARM RAISE", visibility
    return "ARM DOWN", visibility


def draw_pose_skeleton(frame: np.ndarray, kp: np.ndarray) -> None:
    h, w = frame.shape[:2]
    side = max(h, w)                      
    pad_x = (side - w) / 2.0
    pad_y = (side - h) / 2.0

    def xy(point: np.ndarray) -> tuple[int, int]:
        y, x = float(point[0]), float(point[1])
        return int(x * side - pad_x), int(y * side - pad_y)

    links = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]
    for a, b in links:
        if kp[a, 2] >= 0.25 and kp[b, 2] >= 0.25:
            cv2.line(frame, xy(kp[a]), xy(kp[b]), CYAN, 2, cv2.LINE_AA)
    for point in kp:
        if point[2] >= 0.25:
            cv2.circle(frame, xy(point), 4, GREEN, -1, cv2.LINE_AA)



def get_input_device() -> tuple[int | None, int]:
    try:
        default_device = sd.default.device[0]
        if default_device is not None and int(default_device) >= 0:
            info = sd.query_devices(int(default_device), "input")
            device_id: int | None = int(default_device)
        else:
            info = sd.query_devices(kind="input")
            device_id = None
        if int(info.get("max_input_channels", 0)) < 1:
            raise RuntimeError("Selected default audio device has no input channels.")
        native_rate = int(round(float(info.get("default_samplerate", 48000)))) or 48000
        return device_id, native_rate
    except Exception as exc:
        raise RuntimeError(f"No usable microphone/input device found: {exc}") from exc


def audio_worker(interpreter: Any, labels: list[str], stop_event: threading.Event,
                 state: dict[str, Any], lock: threading.Lock) -> None:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    window = int(inp["shape"][1])          

    try:
        device_id, native_rate = get_input_device()
        
        try:
            sd.check_input_settings(device=device_id, samplerate=TM_AUDIO_RATE, channels=1)
            rate = TM_AUDIO_RATE
        except Exception:
            rate = native_rate
        name = sd.query_devices(device_id, "input").get("name", "Default microphone")
        with lock:
            state.update(label="Listening...", score=0.0, error=None, rms=0.0,
                         sample_rate=rate, device=name)
        print(f"[AUDIO] Microphone '{name}' ready at {rate} Hz (model needs {TM_AUDIO_RATE} Hz).")
    except Exception as exc:
        with lock:
            state.update(label="MICROPHONE ERROR", score=0.0, error=str(exc))
        print(f"[AUDIO ERROR] {exc}")
        return

    n_record = int(np.ceil(window * rate / TM_AUDIO_RATE)) + 1
    while not stop_event.is_set():
        try:
            rec = sd.rec(n_record, samplerate=rate, channels=1, dtype="float32",
                         device=device_id, blocking=True)
            x = np.nan_to_num(np.asarray(rec, dtype=np.float32).reshape(-1))
            x = np.clip(x, -1.0, 1.0)
            if rate != TM_AUDIO_RATE:
                g = gcd(rate, TM_AUDIO_RATE)
                x = resample_poly(x, TM_AUDIO_RATE // g, rate // g).astype(np.float32)
            x = x[:window]
            if x.size < window:
                x = np.pad(x, (0, window - x.size))
            rms = float(np.sqrt(np.mean(np.square(x))))

            interpreter.set_tensor(inp["index"], x.reshape(inp["shape"]).astype(np.float32))
            interpreter.invoke()
            scores = interpreter.get_tensor(out["index"])[0]
            best = int(np.argmax(scores))
            label = labels[best] if best < len(labels) else str(best)
            with lock:
                state.update(label=label, score=float(scores[best]), error=None,
                             rms=rms, updated=time.time(), sample_rate=rate)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            with lock:
                state.update(label="AUDIO ERROR", score=0.0, error=message, updated=time.time())
            print(f"[AUDIO ERROR] {message}")
            if stop_event.wait(1.0):
                break



def put_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.6,
             color: tuple[int, int, int] = TEXT, thickness: int = 1) -> None:
    cv2.putText(img, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_card(canvas: np.ndarray, box: tuple[int, int, int, int], title: str,
              value: str, detail: str, score: float | None = None,
              color: tuple[int, int, int] = CYAN) -> None:
    x, y, w, h = box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (64, 79, 101), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + 5, y + h), color, -1)
    put_text(canvas, title.upper(), (x + 20, y + 30), 0.49, MUTED, 1)
    put_text(canvas, value[:28], (x + 20, y + 70), 0.82, TEXT, 2)
    put_text(canvas, detail[:43], (x + 20, y + 101), 0.43, MUTED, 1)
    if score is not None:
        bx, by, bw, bh = x + 20, y + h - 25, w - 40, 8
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (57, 68, 86), -1)
        fill = int(bw * max(0.0, min(1.0, score)))
        cv2.rectangle(canvas, (bx, by), (bx + fill, by + bh), color, -1)


def make_dashboard(camera_frame: np.ndarray, image_label: str, image_score: float,
                   image_runner_up: str, image_runner_score: float,
                   pose_label: str, pose_quality: float,
                   audio: dict[str, Any], fps: float) -> np.ndarray:
    canvas = np.full((DASH_H, DASH_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (DASH_W, 78), (27, 35, 49), -1)
    put_text(canvas, "MULTI-MODAL MONITOR", (28, 40), 0.82, TEXT, 2)
    put_text(canvas, "IMAGE  |  AUDIO  |  POSE", (360, 38), 0.48, CYAN, 1)
    put_text(canvas, f"LIVE  |  {fps:.1f} FPS", (1090, 38), 0.46, GREEN, 1)

    vx, vy, vw, vh = 24, 88, 780, 570
    cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), PANEL, -1)
    fh, fw = camera_frame.shape[:2]
    scale = min((vw - 20) / fw, (vh - 58) / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(camera_frame, (nw, nh), interpolation=cv2.INTER_AREA)
    px = vx + (vw - nw) // 2
    py = vy + 42 + (vh - 52 - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    put_text(canvas, "CAMERA FEED  |  Keep your hand inside the green square",
             (vx + 18, vy + 28), 0.47, MUTED, 1)

    
    crop_side = int(min(nw, nh) * ROI_FRAC)
    cx1 = px + (nw - crop_side) // 2
    cy1 = py + (nh - crop_side) // 2
    cv2.rectangle(canvas, (cx1, cy1), (cx1 + crop_side, cy1 + crop_side), GREEN, 2)

    sx, sw = 828, 428
    draw_card(canvas, (sx, 88, sw, 170), "Image classifier",
              image_label, f"{image_score:.1%} confidence  |  Next: {image_runner_up} {image_runner_score:.0%}",
              image_score, GREEN if image_score >= IMAGE_MIN_CONF else ORANGE)

    audio_label = str(audio.get("label", "Waiting for microphone"))
    audio_score = float(audio.get("score", 0.0))
    audio_error = audio.get("error")
    if audio_error:
        audio_detail = str(audio_error).replace("\n", " ")[:40]
        audio_color = RED
    else:
        rms = float(audio.get("rms", 0.0))
        rate = audio.get("sample_rate", "-")
        audio_detail = f"{audio_score:.1%} score  |  level {rms:.3f}  |  {rate} Hz"
        audio_color = GREEN if audio.get("updated") else ORANGE
    draw_card(canvas, (sx, 278, sw, 170), "Audio classifier",
              audio_label, audio_detail, None if audio_error else audio_score, audio_color)

    draw_card(canvas, (sx, 468, sw, 170), "Pose detector - MoveNet",
              pose_label, f"Landmark visibility: {pose_quality:.1%}  |  Rule-based label",
              pose_quality, CYAN)

    cv2.rectangle(canvas, (24, 674), (1256, 704), (27, 35, 49), -1)
    put_text(canvas, "Q / ESC: STOP     S: SAVE SCREENSHOT     Closing the window with X also stops",
             (40, 695), 0.46, TEXT, 1)
    return canvas


def terminal_quit_pressed() -> bool:
    """Allow Q in the terminal as well as in the OpenCV window (Windows only)."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch().lower() == "q"
    except Exception:
        pass
    return False


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = writer = None
    audio_thread: threading.Thread | None = None
    stop_event = threading.Event()
    audio_state: dict[str, Any] = {
        "label": "Starting microphone...", "score": 0.0, "error": None,
        "rms": 0.0, "updated": None,
    }
    state_lock = threading.Lock()
    last_dashboard: np.ndarray | None = None
    exit_reason = "User stopped the application."

    try:
        image_model, image_labels, audio_model, audio_labels, movenet = load_models()
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam. Close Camera/Teams/Zoom and try CAMERA_INDEX = 1.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam opened but did not return a frame.")

        writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"),
                                 RECORD_FPS, (DASH_W, DASH_H))
        if not writer.isOpened():
            raise RuntimeError("Could not create output/b4_combined.mp4.")

        audio_thread = threading.Thread(
            target=audio_worker,
            args=(audio_model, audio_labels, stop_event, audio_state, state_lock),
            name="B4-Audio-Worker", daemon=True,
        )
        audio_thread.start()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, DASH_W, DASH_H)
        image_history: deque[np.ndarray] = deque(maxlen=IMAGE_SMOOTHING_FRAMES)
        pose_history: deque[str] = deque(maxlen=POSE_SMOOTHING_FRAMES)
        fps_history: deque[float] = deque(maxlen=20)
        previous_time = time.monotonic()
        print("\nB4 dashboard is running. Press Q or ESC to stop.")
        print(f"Recording: {VIDEO_PATH}")

        while True:
            frame = cv2.flip(frame, 1)

           
            probs = predict_image(image_model, frame)
            image_history.append(probs)
            average = np.mean(np.stack(image_history, axis=0), axis=0)
            order = np.argsort(average)[::-1]
            best = int(order[0])
            second = int(order[1]) if len(order) > 1 else best
            image_score = float(average[best])
            image_label = image_labels[best] if image_score >= IMAGE_MIN_CONF else "Uncertain - show hand clearly"
            runner_label = image_labels[second] if second != best else "-"
            runner_score = float(average[second]) if second != best else 0.0

           
            keypoints = run_movenet(movenet, frame)
            raw_pose, pose_quality = classify_pose(keypoints)
            pose_history.append(raw_pose)
            pose_label = Counter(pose_history).most_common(1)[0][0]
            draw_pose_skeleton(frame, keypoints)

            with state_lock:
                audio_snapshot = dict(audio_state)

            now = time.monotonic()
            delta = now - previous_time
            previous_time = now
            if delta > 0:
                fps_history.append(1.0 / delta)
            fps = float(np.mean(fps_history)) if fps_history else 0.0

            dashboard = make_dashboard(frame, image_label, image_score, runner_label, runner_score,
                                       pose_label, pose_quality, audio_snapshot, fps)
            writer.write(dashboard)
            cv2.imshow(WINDOW, dashboard)
            last_dashboard = dashboard

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or terminal_quit_pressed():
                exit_reason = "Quit key pressed."
                break
            if key in (ord("s"), ord("S")):
                cv2.imwrite(str(SCREENSHOT_PATH), dashboard)
                print(f"Screenshot saved: {SCREENSHOT_PATH}")

            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "Dashboard window closed."
                    break
            except cv2.error:
                exit_reason = "Dashboard window closed."
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                exit_reason = "Webcam frame read failed."
                break

    except KeyboardInterrupt:
        exit_reason = "Ctrl+C pressed."
    except Exception as exc:
        exit_reason = f"Application error: {exc}"
        print(f"\n[B4 ERROR] {exc}")

    finally:
        stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
        if audio_thread is not None:
            audio_thread.join(timeout=5.0)
            if audio_thread.is_alive():
                print("[WARN] Audio thread did not stop in time; it will end with the process.")

    if last_dashboard is not None:
        try:
            cv2.imwrite(str(SCREENSHOT_PATH), last_dashboard)
        except Exception:
            pass
    print(f"\n{exit_reason}")
    if VIDEO_PATH.exists():
        print(f"Video saved: {VIDEO_PATH}")
    print("Camera released; B4 application finished.")
    return 0 if last_dashboard is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
