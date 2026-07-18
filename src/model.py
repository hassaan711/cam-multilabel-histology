"""
model.py
--------
DenseNet169 multi-label classification head used for both PanNuke and MoNuSAC.

Architecture
    DenseNet169 (ImageNet pre-trained)
    → Identity (removes default classifier)
    → Linear(1664 → 512) + BatchNorm1d + ReLU + Dropout(p)
    → Linear(512 → num_classes)

Training
    BCEWithLogitsLoss with per-class pos_weight
    AdamW: lr_head=1e-4, lr_backbone=1e-5, weight_decay=1e-4
    CosineAnnealingLR, AMP, grad_clip=1.0
    Batch=16, Epochs=30, sigmoid_threshold=0.50
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DenseNet169MultiLabel(nn.Module):
    """DenseNet169 backbone with a custom multi-label classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
        PanNuke = 5 (Neoplastic, NonneoplasticEpi, Inflammatory, Connective, Dead)
        MoNuSAC = 4 (Epithelial, Lymphocyte, Macrophage, Neutrophil)
    dropout_p : float
        Dropout probability in the classification head.
    """

    def __init__(self, num_classes: int = 5, dropout_p: float = 0.3) -> None:
        super().__init__()
        backbone = models.densenet169(weights=None)
        in_features = backbone.classifier.in_features  # 1664 for DenseNet169
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def load_checkpoint(
    path: str,
    num_classes: int,
    dropout_p: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> DenseNet169MultiLabel:
    """Load a trained DenseNet169MultiLabel from a checkpoint file.

    Parameters
    ----------
    path        : path to .pth checkpoint saved as
                  {'epoch': int, 'model_state': dict, 'val_auc': float}
    num_classes : must match the checkpoint's output dimension
    dropout_p   : must match training configuration
    device      : target device

    Returns
    -------
    model in eval mode on the specified device
    """
    model = DenseNet169MultiLabel(num_classes=num_classes, dropout_p=dropout_p)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(
        f"Loaded checkpoint: epoch={ckpt['epoch']}, "
        f"val_auc={ckpt['val_auc']:.4f}"
    )
    return model
