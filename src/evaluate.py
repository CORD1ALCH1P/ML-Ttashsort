"""Evaluate a trained checkpoint on the held-out test split.

Produces: accuracy, per-class precision/recall/F1, confusion matrix (png + csv),
CPU inference latency (ms/image, single-image batch), and model size on disk.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_dataloaders
from models import ARCHITECTURES
from inference import load_checkpoint

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@torch.no_grad()
def evaluate_architecture(arch: str, batch_size: int = 32, n_speed_runs: int = 50):
    model, classes, img_size = load_checkpoint(arch)
    _, _, test_loader, ds_classes = build_dataloaders(batch_size=batch_size, img_size=img_size)
    assert ds_classes == classes

    all_preds, all_labels, all_probs = [], [], []
    for images, labels in test_loader:
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.max(1).values.tolist())

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix: {arch}")
    plt.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{arch}.png", dpi=120)
    plt.close()

    # CPU inference latency: single-image batches, warm-up then measure
    dummy = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):
        model(dummy)
    t0 = time.time()
    for _ in range(n_speed_runs):
        model(dummy)
    latency_ms = (time.time() - t0) / n_speed_runs * 1000

    model_size_mb = (MODELS_DIR / f"{arch}.pt").stat().st_size / (1024 * 1024)
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "arch": arch,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {c: report[c] for c in classes},
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "latency_ms_per_image": latency_ms,
        "model_size_mb": model_size_mb,
        "n_params": n_params,
        "img_size": img_size,
    }

    with open(REPORTS_DIR / f"eval_{arch}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{arch}] acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
          f"latency={latency_ms:.2f}ms size={model_size_mb:.1f}MB params={n_params/1e6:.2f}M")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHITECTURES + ["all"], default="all")
    args = parser.parse_args()

    archs = ARCHITECTURES if args.arch == "all" else [args.arch]
    for arch in archs:
        evaluate_architecture(arch)


if __name__ == "__main__":
    main()
