import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)


load_dotenv()

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

project_folder = Path(__file__).resolve().parent
image_path = project_folder / "test_background.jpeg"

device = "cuda" if torch.cuda.is_available() else "cpu"
token = os.getenv("HF_TOKEN")

print(f"Using device: {device}")
print("Loading SegFormer...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    token=token,
)

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    token=token,
)

model.to(device)
model.eval()

print("SegFormer loaded successfully!")

image = Image.open(image_path).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt",
).to(device)

print("Detecting surfaces...")

with torch.inference_mode():
    outputs = model(**inputs)

resized_logits = torch.nn.functional.interpolate(
    outputs.logits,
    size=(image.height, image.width),
    mode="bilinear",
    align_corners=False,
)

class_map = resized_logits.argmax(dim=1)[0].cpu().numpy()

detected_ids = np.unique(class_map)
detected_labels = [
    model.config.id2label[int(class_id)]
    for class_id in detected_ids
]

print("Detected labels:")
print(detected_labels)

surface_names = {
    "table",
    "floor",
    "counter",
    "countertop",
    "desk",
}

surface_ids = [
    int(class_id)
    for class_id, label in model.config.id2label.items()
    if label.lower() in surface_names
]

surface_mask = np.isin(class_map, surface_ids)

if not surface_mask.any():
    print("No supported surface was detected.")

else:
    original_array = np.array(image).astype(np.float32)

    green_overlay = np.zeros_like(original_array)
    green_overlay[:, :, 1] = 255

    preview_array = original_array.copy()

    preview_array[surface_mask] = (
        original_array[surface_mask] * 0.55
        + green_overlay[surface_mask] * 0.45
    )

    preview = Image.fromarray(
        preview_array.astype(np.uint8)
    )

    output_path = (
        project_folder / "surface_mask_preview.png"
    )

    preview.save(output_path)

    print(f"Surface preview saved to: {output_path}")