# CONCEPTS — Teachable Machine and Transfer Learning

> Working notes for Task B4. Complete the comparison with your Project A implementation and the measured model summaries before submission.

## 1. What Teachable Machine does

Teachable Machine lets a user collect examples for custom classes, trains a classifier using a pre-trained feature-extraction pipeline, and exports a model for inference. The exact architecture and preprocessing depend on the project type (image, audio, or pose) and the export format.

## 2. Transfer learning by feature extraction

A pre-trained network has already learned useful representations from a large dataset. In a feature-extraction workflow, the pre-trained part converts each input into a feature representation, while a smaller task-specific classification head learns to separate the user's custom classes.

For the image model, explain how convolutional layers represent visual patterns such as edges, textures, and object parts. Confirm the architecture and which parameters are trainable by inspecting the actual exported model rather than assuming every Teachable Machine export has identical internals.

## 3. Comparison with Project A — Task A6.2

Add a direct comparison after reviewing the Project A code:

| Aspect | Project A manual feature extraction | Teachable Machine |
|---|---|---|
| Feature extractor | Fill from Project A | Verify from exported model |
| Custom classifier | Fill from Project A | Inspect exported model |
| Data preparation | Fill from Project A | Project-specific preprocessing |
| Training workflow | Manual code | Teachable Machine interface |

## 4. Limitations

Discuss the limitations observed in your experiments, such as class imbalance, insufficient or non-diverse training examples, lighting/background sensitivity, confusion between similar classes, latency, and the limits of a small custom dataset. Do not claim a model is production-ready based only on its training Preview accuracy.

## 5. Model introspection results

Fill this table using `python inspect_models.py` after all three exports are present.

| Model | Input shape | Output shape | Total parameters | Trainable parameters | Layer count |
|---|---|---|---:|---:|---:|
| Image | To be measured | To be measured | To be measured | To be measured | To be measured |
| Audio | To be measured | To be measured | To be measured | To be measured | To be measured |
| Pose | To be measured | To be measured | To be measured | To be measured | To be measured |
