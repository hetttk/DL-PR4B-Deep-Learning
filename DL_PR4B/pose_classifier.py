"""B3 starter — pose classifier.

This file is intentionally a scaffold. Inspect the exported pose model's real
input shape and determine the matching PoseNet/MoveNet preprocessing first.
Do not assume raw (17, 2) coordinates are the model input without verification.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "pose"


def main() -> int:
    model_path = MODEL_DIR / "keras_model.h5"
    labels_path = MODEL_DIR / "labels.txt"
    missing = [p.name for p in (model_path, labels_path) if not p.exists()]
    if missing:
        print("B3 is not ready. Add the pose export files to models/pose/: " + ", ".join(missing))
        return 1
    print("Pose export found. Next step: inspect model input and matching pose-estimation pipeline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
