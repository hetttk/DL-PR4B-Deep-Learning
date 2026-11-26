"""B4.4 model introspection for all model files actually supplied."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

ROOT = Path(__file__).resolve().parent


class TeachableMachineDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def count_weights(weights):
    return int(sum(np.prod(weight.shape) for weight in weights))


def recursive_layer_count(layer):
    """Count nested Keras layers in nested Sequential/Functional models."""
    nested = getattr(layer, "layers", None)
    if not nested:
        return 1
    return 1 + sum(recursive_layer_count(child) for child in nested)


def main():
    image_path = ROOT / "models" / "image" / "keras_model.h5"
    audio_path = ROOT / "models" / "audio" / "soundclassifier_with_metadata.tflite"
    pose_json = ROOT / "models" / "pose" / "web" / "model.json"

    print("\n[IMAGE] Keras H5 model")
    if image_path.exists():
        model = tf.keras.models.load_model(
            str(image_path), compile=False,
            custom_objects={"DepthwiseConv2D": TeachableMachineDepthwiseConv2D},
        )
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
        print("Top-level layers:", len(model.layers))
        print("Recursive layers (including nested models):", recursive_layer_count(model))
        print("Total parameters:", model.count_params())
        print("Trainable parameters:", count_weights(model.trainable_weights))
        print("Non-trainable parameters:", count_weights(model.non_trainable_weights))
        model.summary()
    else:
        print("Not found:", image_path)

    print("\n[AUDIO] TensorFlow Lite AudioClassifier model")
    if audio_path.exists():
        print("File:", audio_path)
        print("Format: .tflite (not .h5)")
        print("The model is loaded by MediaPipe Tasks AudioClassifier in combined_app.py.")
        print("Keras model.summary()/trainable_weights are not available for this TFLite export.")
    else:
        print("Not found:", audio_path)

    print("\n[POSE B3] TensorFlow.js export")
    if pose_json.exists():
        with pose_json.open("r", encoding="utf-8") as handle:
            model_json = json.load(handle)
        print("File:", pose_json)
        print("Format: TensorFlow.js; this is not a Keras .h5 file.")
        print("Model format/architecture fields:", model_json.get("format"), model_json.get("modelTopology", {}).get("model_config", {}).get("class_name", "not present"))
        print("Python B4 instead loads MoveNet SinglePose Lightning from TensorFlow Hub.")
    else:
        print("Not found:", pose_json)

    print("\nFor the project as supplied, the image model is the only Keras H5 model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
