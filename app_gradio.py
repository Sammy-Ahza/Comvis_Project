import json
import os
from typing import Tuple

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

MODELS_DIR = "/home/semi/Vscode/Comvis/models"
MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv3_ham10000.pt")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.json")


def load_model() -> Tuple[nn.Module, dict]:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(MAPPING_PATH)):
        raise FileNotFoundError("Model or class mapping not found. Train the model first.")

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(idx_to_class))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, idx_to_class


MODEL, IDX_TO_CLASS = load_model()

TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]
    results = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))}
    top_class = max(results, key=results.get)
    return top_class, results


def build_interface():
    title = "Skin Cancer Classification (MobileNetV3)"
    desc = "Upload a dermoscopic image (HAM10000-like). The model predicts 7 skin lesion classes."

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=1, label="Prediction"), gr.Label(label="Class probabilities")],
        title=title,
        description=desc,
        allow_flagging="never",
    )
    return demo


def main():
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
