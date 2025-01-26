import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from PIL import Image

import config
from Models import VisionModel

_model_path = config.config.get("model_path", "./models")
THRESHOLD = config.config.get("threshold", 0.4)

device = config.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
_use_cpu = device == "cpu"

model = VisionModel.load_model(_model_path)
model.eval()
model = model.to(device)

with open(Path(_model_path) / "top_tags.txt", "r") as f:
    top_tags = [line.strip() for line in f.readlines() if line.strip()]


async def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(
        image_tensor,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    return image_tensor


@torch.no_grad()
async def predict(image: Image.Image) -> tuple[list[str], dict[str, Any]]:
    image_tensor = await prepare_image(image, model.image_size)
    batch = {
        "image": image_tensor.unsqueeze(0).to(device),
    }

    with torch.amp.autocast_mode.autocast(device, enabled=not _use_cpu):
        preds = model(batch)
        tag_preds = preds["tags"].sigmoid().cpu()

    scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
    predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]

    return predicted_tags, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to image file")
    args = parser.parse_args()
    print(f"Using device: {device}")
    image = Image.open(args.image)
    loop = asyncio.new_event_loop()
    time_start = time.time()
    tag_string, scores = loop.run_until_complete(predict(image))
    print(f"Predicted tags: {tag_string}")
    for tag, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{tag}: {score:.3f}")
    loop.close()
    print(f"Time cost: {time.time() - time_start:.3f}s")
