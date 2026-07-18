"""
preprocessing.py
----------------
Image preprocessing transforms and inference utilities shared across
PanNuke and MoNuSAC experiments.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# ImageNet statistics used for DenseNet169 pre-training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE  = 256    # model training resolution
THRESHOLD = 0.50   # sigmoid threshold for binary prediction
NOISE_THR = 0.25   # CAM visualisation noise floor
ALPHA_SCALE = 0.85 # CAM overlay opacity


# ── Transforms ────────────────────────────────────────────────────────────────

_transform_256 = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_transform_native = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess(
    img_rgb: np.ndarray,
    requires_grad: bool = False,
    native: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert a HWC uint8 RGB image to a (1, 3, H, W) float32 tensor.

    Parameters
    ----------
    img_rgb       : (H, W, 3) uint8 numpy array
    requires_grad : set True when using gradient-based CAM methods
    native        : if True, skip the 256×256 resize (for resolution ablation)
    device        : target device

    Returns
    -------
    (1, 3, H, W) float32 tensor on device
    """
    tfm = _transform_native if native else _transform_256
    t   = tfm(img_rgb).unsqueeze(0).to(device)
    return t.requires_grad_(True) if requires_grad else t


@torch.no_grad()
def predict(
    img_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    threshold: float = THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return binary labels and sigmoid probabilities.

    Parameters
    ----------
    img_rgb   : (H, W, 3) uint8 RGB image
    model     : trained DenseNet169MultiLabel in eval mode
    device    : device the model is on
    threshold : sigmoid threshold for binary prediction

    Returns
    -------
    (pred_labels, probs) both shape (num_classes,)
    """
    t      = preprocess(img_rgb, requires_grad=False, device=device)
    logits = model(t)
    probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (probs >= threshold).astype(np.uint8), probs


# ── Mask and overlay utilities ────────────────────────────────────────────────

def resize_mask_channel(
    mask: np.ndarray,          # (H, W) single-channel mask
    hw: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
) -> np.ndarray:               # (H_out, W_out) float32
    """Resize a single mask channel to the target (H, W) using nearest-neighbour."""
    ch = mask.astype(np.float32)
    if ch.shape == hw:
        return ch
    return cv2.resize(ch, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def build_cam_overlay(
    cam: np.ndarray,           # (H, W) float32 [0, 1]
    class_color: Tuple[float, float, float],  # (r, g, b) in [0, 1]
    orig_img: np.ndarray,      # (H, W, 3) uint8
    noise_thr: float = NOISE_THR,
    alpha_scale: float = ALPHA_SCALE,
) -> np.ndarray:               # (H, W, 3) uint8
    """Overlay a CAM heatmap on the original image using the class colour.

    Pixels below noise_thr are zeroed (suppressed) to reduce clutter.

    Parameters
    ----------
    cam          : normalised CAM heatmap in [0, 1]
    class_color  : (r, g, b) tuple in [0, 1] for the target class
    orig_img     : original RGB image at the same spatial resolution as cam
    noise_thr    : CAM values below this are set to 0
    alpha_scale  : maximum opacity of the CAM overlay

    Returns
    -------
    Composited RGB image as uint8
    """
    r, g, b = class_color
    clean   = np.where(cam >= noise_thr, cam, 0.0)
    comp_c  = np.stack([clean * r, clean * g, clean * b], axis=-1)
    alpha   = (clean * alpha_scale)[..., None]
    bg      = orig_img.astype(np.float32) / 255.0
    out     = (comp_c * alpha + bg * (1 - alpha)).clip(0, 1)
    return (out * 255).astype(np.uint8)


def build_composite_overlay(
    cams: dict,                # {class_idx: cam_array}
    class_colors: list,        # list of (r,g,b) tuples
    orig_img: np.ndarray,
    noise_thr: float = NOISE_THR,
    alpha_scale: float = ALPHA_SCALE,
) -> np.ndarray:
    """Max-blend multiple class CAMs into a single composite overlay.

    For each pixel, the class with the highest CAM activation wins,
    preventing colour mixing that would make the overlay unreadable.

    Parameters
    ----------
    cams         : dict mapping class_idx → (H, W) float32 CAM
    class_colors : list of (r, g, b) tuples indexed by class_idx
    orig_img     : (H, W, 3) uint8 original image
    noise_thr    : suppress pixels below this CAM value
    alpha_scale  : maximum overlay opacity

    Returns
    -------
    (H, W, 3) uint8 composite overlay
    """
    if not cams:
        return orig_img.copy()

    # Use the shape of the first CAM
    h, w = next(iter(cams.values())).shape
    comp_alpha  = np.zeros((h, w), dtype=np.float32)
    comp_colour = np.zeros((h, w, 3), dtype=np.float32)

    for c_idx, cam in cams.items():
        r2, g2, b2 = class_colors[c_idx]
        clean      = np.where(cam >= noise_thr, cam, 0.0)
        layer_a    = clean * alpha_scale
        brighter   = layer_a > comp_alpha
        comp_alpha[brighter]     = layer_a[brighter]
        comp_colour[brighter, 0] = clean[brighter] * r2
        comp_colour[brighter, 1] = clean[brighter] * g2
        comp_colour[brighter, 2] = clean[brighter] * b2

    bg  = orig_img.astype(np.float32) / 255.0
    out = (comp_colour * comp_alpha[..., None] + bg * (1 - comp_alpha[..., None])).clip(0, 1)
    return (out * 255).astype(np.uint8)
