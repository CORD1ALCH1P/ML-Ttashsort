"""Shared inference helpers used by both evaluate.py and the Streamlit web app."""
import time
from pathlib import Path

import torch
from PIL import Image

from dataset import build_transforms
from models import build_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_checkpoint(arch: str):
    ckpt = torch.load(MODELS_DIR / f"{arch}.pt", map_location="cpu")
    model, _ = build_model(arch, len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], ckpt["img_size"]


def available_architectures():
    return sorted(p.stem for p in MODELS_DIR.glob("*.pt"))


@torch.no_grad()
def predict(model, classes, img_size: int, image: Image.Image):
    """Run inference on a single PIL image. Returns (probs_dict, latency_ms)."""
    _, eval_tf = build_transforms(img_size)
    tensor = eval_tf(image.convert("RGB")).unsqueeze(0)

    t0 = time.time()
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.time() - t0) * 1000

    probs_dict = {cls: float(p) for cls, p in zip(classes, probs.tolist())}
    return probs_dict, latency_ms
