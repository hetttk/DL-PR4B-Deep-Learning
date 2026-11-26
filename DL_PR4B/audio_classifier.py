import json
import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

try:  
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "audio"
MODEL_PATH = MODEL_DIR / "soundclassifier_with_metadata.tflite"
LABELS_PATH = MODEL_DIR / "labels.txt"
METADATA_PATH = MODEL_DIR / "metadata.json"

SAMPLE_RATE = 44100  #


def load_labels(path: Path) -> list[str]:
    """labels.txt lines look like '0 Background Noise'."""
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            labels.append(line.split(" ", 1)[1] if line[0].isdigit() else line)
    return labels


def load_audio(path: str) -> np.ndarray:
    """Read a wav file as mono float32 in [-1, 1] at 44.1 kHz."""
    rate, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return data


def classify(interpreter, labels, samples: np.ndarray):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], samples.reshape(inp["shape"]).astype(np.float32))
    interpreter.invoke()
    scores = interpreter.get_tensor(out["index"])[0]
    best = int(np.argmax(scores))
    return labels[best], float(scores[best]), scores


def listen(interpreter, labels, window: int) -> int:
    """Classify the microphone input, one ~1 second window at a time."""
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        print("Microphone mode needs the 'sounddevice' package.")
        print("Install it with:  pip install sounddevice")
        return 1

    print("Listening... make a sound (clap, knock, whistle). Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while True:
                data, _ = stream.read(window)
                label, conf, _ = classify(interpreter, labels, data[:, 0])
                print(f"{label:<16} {conf:6.1%}")
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


def main() -> int:
    missing = [p.name for p in (MODEL_PATH, LABELS_PATH, METADATA_PATH) if not p.exists()]
    if missing:
        print("B2 is not ready. Add the audio export files to models/audio/: " + ", ".join(missing))
        print(f"Looking in: {MODEL_DIR}")
        if MODEL_DIR.exists():
            print("Found there:", [p.name for p in MODEL_DIR.iterdir()] or "(empty folder)")
        else:
            print("That folder does not exist.")
        return 1
    labels = load_labels(LABELS_PATH)
    classes = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("classes", [])
    if classes and classes != labels:
        print("Warning: labels.txt and metadata.json classes differ.")

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    window = int(interpreter.get_input_details()[0]["shape"][1]) 

    if len(sys.argv) < 2:
        return listen(interpreter, labels, window)

    audio = load_audio(sys.argv[1])
    if len(audio) < window:  
        audio = np.pad(audio, (0, window - len(audio)))

    hop = window // 2
    for start in range(0, len(audio) - window + 1, hop):
        label, conf, scores = classify(interpreter, labels, audio[start:start + window])
        t = start / SAMPLE_RATE
        print(f"{t:5.2f}s  {label:<16} {conf:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
