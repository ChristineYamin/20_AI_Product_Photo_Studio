import numpy as np
from pathlib import Path
from PIL import Image


import os
import torch
from dotenv import load_dotenv
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

load_dotenv()

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"
token = os.getenv("HF_TOKEN")

print(f"Using device: {device}")
print("Loading Depth ANything V2...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    token=token,
)

model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_NAME,
    token=token,
)   

model.to(device)
model.eval()

print("Depth Anything V2 loaded successfully!")

project_folder = Path(__file__).resolve().parent
image_path = project_folder / "test_background.jpg"

image = Image.open(image_path).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt",
).to(device)

print("Estimating depth...")

with torch.inference_mode():
    outputs = model(**inputs)

# Resize the depth prediction to match the original image.
depth_tensor = torch.nn.functional.interpolate(
    outputs.predicted_depth.unsqueeze(1),
    size=(image.height, image.width),
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_array = depth_tensor.cpu().numpy()

# Normalize for display only—not distances in metres.
depth_min = float(depth_array.min())
depth_max = float(depth_array.max())

normalized_depth = (
    (depth_array - depth_min)
    / max(depth_max - depth_min, 1e-8)
)

depth_preview = Image.fromarray(
    (normalized_depth * 255).astype(np.uint8)
)

output_path = project_folder / "test_depth_map.png"
depth_preview.save(output_path)

print(f"Depth map saved to: {output_path}")

# Save the original depth values for our placement test.
np.save(
    project_folder / "test_depth_values.npy",
    depth_array,
)

print("Raw depth values saved!")

# Examine depth across the central part of the image.
height, width = depth_array.shape

central_depth = depth_array[
    :,
    int(width * 0.35):int(width * 0.65),
]

row_depth = np.median(central_depth, axis=1)

# Smooth small changes in the depth prediction.
window_size = 15
padding = window_size // 2

smoothed_depth = np.convolve(
    np.pad(row_depth, (padding, padding), mode="edge"),
    np.ones(window_size) / window_size,
    mode="valid",
)

depth_change = np.gradient(smoothed_depth)

# Search for a possible rear surface boundary.
search_start = int(height * 0.55)
search_end = int(height * 0.90)

boundary_y = search_start + int(
    np.argmax(depth_change[search_start:search_end])
)

suggested_y = min(
    height - 1,
    boundary_y + int(height * 0.10),
)

# Draw the suggested product contact point.
from PIL import ImageDraw

placement_preview = image.copy()
draw = ImageDraw.Draw(placement_preview)

center_x = width // 2
radius = 12

draw.ellipse(
    (
        center_x - radius,
        suggested_y - radius,
        center_x + radius,
        suggested_y + radius,
    ),
    fill="red",
)

placement_preview.save(
    project_folder / "test_surface_preview.png"
)

print(
    f"Suggested product bottom: "
    f"{suggested_y / height:.0%} of image height"
)