"""Fine-tune one architecture on the waste-classification dataset (CPU-friendly).

Two-phase training:
  Phase 1 (head warmup):   backbone frozen, only the new classification head trains.
  Phase 2 (fine-tuning):   whole network unfrozen, differential learning rates
                            (small LR for backbone, larger LR for head).
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import build_dataloaders
from models import build_model, freeze_backbone, unfreeze_all, param_groups, ARCHITECTURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_n += images.size(0)

    return total_loss / total_n, total_correct / total_n


def train_one_architecture(
    arch: str,
    epochs_head: int = 5,
    epochs_finetune: int = 5,
    batch_size: int = 32,
    img_size: int = 160,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = "cpu"

    train_loader, val_loader, _, classes = build_dataloaders(batch_size=batch_size, img_size=img_size)
    num_classes = len(classes)

    model, head_modules = build_model(arch, num_classes, pretrained=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"arch": arch, "epochs": []}
    best_val_acc = 0.0
    best_state = None
    t_start = time.time()

    # Phase 1: head warmup
    freeze_backbone(model, head_modules)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=head_lr)
    for epoch in range(epochs_head):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(f"[{arch}] phase1 epoch {epoch+1}/{epochs_head} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        history["epochs"].append({
            "phase": 1, "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Phase 2: fine-tune whole network with differential LR
    unfreeze_all(model)
    optimizer = torch.optim.Adam(param_groups(model, head_modules, backbone_lr, head_lr))
    for epoch in range(epochs_finetune):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(f"[{arch}] phase2 epoch {epoch+1}/{epochs_finetune} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        history["epochs"].append({
            "phase": 2, "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    total_time = time.time() - t_start
    history["train_time_sec"] = total_time
    history["best_val_acc"] = best_val_acc
    history["classes"] = classes

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    torch.save({"arch": arch, "state_dict": best_state, "classes": classes, "img_size": img_size},
               MODELS_DIR / f"{arch}.pt")

    with open(REPORTS_DIR / f"train_history_{arch}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[{arch}] done in {total_time:.1f}s, best_val_acc={best_val_acc:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHITECTURES + ["all"], default="all")
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()

    archs = ARCHITECTURES if args.arch == "all" else [args.arch]
    for arch in archs:
        train_one_architecture(
            arch,
            epochs_head=args.epochs_head,
            epochs_finetune=args.epochs_finetune,
            batch_size=args.batch_size,
            img_size=args.img_size,
        )


if __name__ == "__main__":
    main()
