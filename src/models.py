"""Model factory: 5 CPU-friendly torchvision architectures fine-tuned for waste classification."""
import torch.nn as nn
from torchvision import models

ARCHITECTURES = [
    "mobilenet_v2",
    "mobilenet_v3_small",
    "resnet18",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
]


def _replace_head(model, arch: str, num_classes: int):
    """Swap the ImageNet head for a fresh num_classes head. Returns the new head module(s)."""
    if arch == "mobilenet_v2":
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
        return [model.classifier[1]]
    if arch == "mobilenet_v3_small":
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, num_classes)
        return [model.classifier[3]]
    if arch == "resnet18":
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return [model.fc]
    if arch == "shufflenet_v2_x1_0":
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return [model.fc]
    if arch == "efficientnet_b0":
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
        return [model.classifier[1]]
    raise ValueError(f"Unknown architecture: {arch}")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None

    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
    elif arch == "resnet18":
        model = models.resnet18(weights=weights)
    elif arch == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=weights)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    head_modules = _replace_head(model, arch, num_classes)
    return model, head_modules


def freeze_backbone(model, head_modules):
    """Freeze all parameters except those belonging to the head modules."""
    head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_param_ids


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def param_groups(model, head_modules, backbone_lr: float, head_lr: float):
    """Differential learning rates: small LR for backbone, larger for the new head."""
    head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_param_ids]
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
