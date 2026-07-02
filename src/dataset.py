"""Dataset / dataloader helpers for the TrashNet split folders."""
from pathlib import Path

import torch
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "split"

IMG_SIZE = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(batch_size: int = 32, img_size: int = IMG_SIZE, num_workers: int = 2):
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_ROOT / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATA_ROOT / "test", transform=eval_tf)

    assert train_ds.classes == val_ds.classes == test_ds.classes, "Class order mismatch across splits"

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_ds.classes
