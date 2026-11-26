import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"        
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       

import time
import cv2
import numpy as np
import tensorflow as tf

try:
    import tf_keras  
except ImportError:
    if tf.__version__ >= "2.16":
        raise SystemExit("Missing package. Run:  pip install tf_keras")

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    HAS_PYAUTOGUI = True
except Exception as e:
    print(f"[warn] pyautogui unavailable ({e}); actions will only be printed.")
    HAS_PYAUTOGUI = False

# ----------------------------- Configuration -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "image", "labels.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SCREENSHOT_PATH = os.path.join(OUTPUT_DIR, "b1_image_result.jpg")

CONF_THRESHOLD = 0.90   
HOLD_FRAMES = 5         # consecutive frames the gesture must be stable
COOLDOWN_SEC = 0.7      # minimum seconds between two actions
ROI_FRAC = 0.6          # hand box = this fraction of frame height (0 = use full frame)

GESTURE_ACTIONS = {
    "thumbsup": "volumeup",
    "thumbsdown": "volumedown",
}
# -------------------------------------------------------------------------


def norm(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def load_labels(path):
    """labels.txt lines look like '1 Thumbsup' -> strip the numeric prefix."""
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            labels.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


def preprocess(frame_bgr):
    """BGR frame -> (1, 224, 224, 3) float32 in [-1, 1]."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(img, axis=0)


def get_roi(frame):
    """Centered square box. Only this region goes to the model, so the hand
    fills the input like the training images (less background / face noise)."""
    if ROI_FRAC <= 0:
        return frame, None
    h, w = frame.shape[:2]
    side = int(h * ROI_FRAC)
    x1, y1 = (w - side) // 2, (h - side) // 2
    return frame[y1:y1 + side, x1:x1 + side], (x1, y1, x1 + side, y1 + side)


def draw_probs(frame, labels, probs):
    """Show every class probability (helps debugging wrong predictions)."""
    h, w = frame.shape[:2]
    for i, (name, p) in enumerate(zip(labels, probs)):
        y = 90 + i * 28
        cv2.putText(frame, f"{name}: {p * 100:5.1f}%", (w - 260, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame


def draw_overlay(frame, label, confidence, last_action):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}  {confidence * 100:.1f}%", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    bar_x, bar_y, bar_w, bar_h = 15, h - 45, w - 30, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 2)
    color = (0, 200, 0) if confidence >= CONF_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * confidence), bar_y + bar_h), color, -1)
    if last_action:
        cv2.putText(frame, f"Last action: {last_action}", (15, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def trigger(action):
    print(f"[action] {action}")
    if HAS_PYAUTOGUI:
        pyautogui.press(action)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)
    print("Classes:", labels)
    print("Model input:", model.input_shape, "output:", model.output_shape)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    stable_label, stable_count = None, 0
    last_trigger_time = 0.0
    last_action = ""
    saved = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view

        roi, box = get_roi(frame)
        prediction = model.predict(preprocess(roi), verbose=0)
        class_idx = int(np.argmax(prediction))
        confidence = float(prediction[0][class_idx])
        label = labels[class_idx] if class_idx < len(labels) else str(class_idx)

        # Require a confident, stable gesture before acting
        if confidence >= CONF_THRESHOLD and label == stable_label:
            stable_count += 1
        elif confidence >= CONF_THRESHOLD:
            stable_label, stable_count = label, 1
        else:
            stable_label, stable_count = None, 0

        action = GESTURE_ACTIONS.get(norm(label))
        now = time.time()
        if (action and stable_count >= HOLD_FRAMES
                and now - last_trigger_time >= COOLDOWN_SEC):
            trigger(action)
            last_action = f"{label} -> {action}"
            last_trigger_time = now

        frame = draw_overlay(frame, label, confidence, last_action)
        frame = draw_probs(frame, labels, prediction[0])
        if box:
            cv2.rectangle(frame, box[:2], box[2:], (255, 255, 255), 2)
        cv2.imshow("B1 - Gesture Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(SCREENSHOT_PATH, frame)
            saved = True
            print(f"Saved {SCREENSHOT_PATH}")
        elif key == ord("q"):
            if not saved:
                cv2.imwrite(SCREENSHOT_PATH, frame)
                print(f"Saved {SCREENSHOT_PATH}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    
