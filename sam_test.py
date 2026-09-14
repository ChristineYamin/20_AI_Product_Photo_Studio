import torch
from transformers import Sam2Model, Sam2Processor


MODEL_NAME = "facebook/sam2.1-hiera-tiny"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("Loading SAM 2 processor...")

processor = Sam2Processor.from_pretrained(
    MODEL_NAME
)

print("Loading SAM 2 model...")

model = Sam2Model.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("SAM 2 loaded successfully!")