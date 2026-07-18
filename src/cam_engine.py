"""
cam_engine.py
-------------
Unified CAM engine implementing all four methods in the 2×2 factorial design:

    ┌──────────────────┬──────────────────────┬───────────────────────────────┐
    │                  │ Single Layer (db4)   │ Feature Pyramid (db1+db2+db4) │
    ├──────────────────┼──────────────────────┼───────────────────────────────┤
    │ GAP gradient     │ Standard GradCAM     │ FPN-GradCAM                   │
    │ Pixel-wise grad  │ Standard LayerCAM    │ FPN-LayerCAM                  │
    └──────────────────┴──────────────────────┴───────────────────────────────┘

The only difference between GradCAM and LayerCAM is how channel importance
weights are computed from the gradient (see _cam_from_acts_grads docstring).

Usage
-----
    from src.cam_engine import CAMEngine, STD_CFG, FPN_CFG

    engine = CAMEngine(model, FPN_CFG, use_layercam=True)   # FPN-LayerCAM
    cam    = engine.generate(tensor, class_idx=0, guide=img_rgb)
    engine.remove_hooks()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Default layer configurations ──────────────────────────────────────────────

STD_CFG: List[Tuple[str, float]] = [
    ("backbone.features.denseblock4", 1.00),
]
"""Single-layer config targeting DenseNet169's deepest dense block."""

FPN_CFG: List[Tuple[str, float]] = [
    ("backbone.features.denseblock1", 0.20),
    ("backbone.features.denseblock2", 0.35),
    ("backbone.features.denseblock4", 0.45),
]
"""Feature Pyramid config: shallow → deep, weighted 0.20 : 0.35 : 0.45."""


# ── CAMEngine ────────────────────────────────────────────────────────────────

class CAMEngine:
    """
    Unified gradient-based CAM engine for all four factorial methods.

    Parameters
    ----------
    model        : trained DenseNet169MultiLabel in eval mode (shared)
    layer_cfg    : list of (dotpath, blend_weight) tuples.
                   One entry  → Standard GradCAM or LayerCAM.
                   Three entries → FPN-GradCAM or FPN-LayerCAM.
    use_layercam : if True, use pixel-wise gradient retention (LayerCAM).
                   if False, use global average pooling of gradients (GradCAM).
    sharpen      : apply joint bilateral sharpening to the fused heatmap.
                   Only active when len(layer_cfg) > 1 (FPN variants).

    Notes
    -----
    call remove_hooks() after all processing to avoid accumulating hooks.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_cfg: List[Tuple[str, float]],
        use_layercam: bool = False,
        sharpen: bool = True,
    ) -> None:
        self.model = model
        self.layer_cfg = layer_cfg
        self.use_layercam = use_layercam
        self.sharpen = sharpen and (len(layer_cfg) > 1)

        self._handles: List = []
        self._store: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
            name: {"act": None, "grad": None} for name, _ in layer_cfg
        }
        self._register_hooks()

    # ── Private ───────────────────────────────────────────────────────────────

    def _sub(self, dotpath: str) -> nn.Module:
        m = self.model
        for k in dotpath.split("."):
            m = getattr(m, k)
        return m

    def _register_hooks(self) -> None:
        for name, _ in self.layer_cfg:
            mod = self._sub(name)
            n = name
            self._handles.append(
                mod.register_forward_hook(
                    lambda m, i, o, n=n:
                        self._store[n].__setitem__("act", o.detach())
                )
            )
            self._handles.append(
                mod.register_full_backward_hook(
                    lambda m, gi, go, n=n:
                        self._store[n].__setitem__("grad", go[0].detach())
                )
            )

    def _cam_from_acts_grads(
        self,
        act: torch.Tensor,   # (1, C, h, w)
        grad: torch.Tensor,  # (1, C, h, w)
        hw: Tuple[int, int],
    ) -> np.ndarray:          # (H, W) float32 in [0, 1]
        """Compute per-layer CAM.

        GradCAM:  α_k = GAP(∂Y^c / ∂A^k)  →  scalar per channel.
                  No spatial gradient information retained. The global average
                  pooling acts as an implicit regulariser, diluting co-occurring
                  class gradients.

        LayerCAM: weights = ReLU(∂Y^c / ∂A^k)  →  spatial map per channel.
                  Only locally-positive gradient positions contribute. This
                  preserves fine-grained spatial evidence but also amplifies
                  multi-label gradient contamination from co-occurring classes.
        """
        if self.use_layercam:
            cam = F.relu(
                (F.relu(grad) * act).sum(dim=1, keepdim=True)  # (1,1,h,w)
            )
        else:
            alpha = grad.mean(dim=(2, 3), keepdim=True)         # (1,C,1,1)
            cam = F.relu(
                (alpha * act).sum(dim=1, keepdim=True)          # (1,1,h,w)
            )

        arr = cam.squeeze().cpu().numpy()
        arr = np.maximum(arr, 0)
        interp = cv2.INTER_CUBIC if self.sharpen else cv2.INTER_LINEAR
        arr = cv2.resize(arr, (hw[1], hw[0]), interpolation=interp)
        if arr.max() > 1e-8:
            arr /= arr.max()
        return arr.astype(np.float32)

    @staticmethod
    def _bilateral_sharpen(
        cam: np.ndarray,    # (H, W) float32 [0, 1]
        guide: np.ndarray,  # (H, W, 3) uint8 RGB
    ) -> np.ndarray:
        """Joint bilateral filter using the H&E image as a guide.

        Prevents smoothing across nucleus boundaries, snapping heatmap
        contours to actual nuclear membrane transitions.
        """
        gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        u8 = (cam * 255).astype(np.uint8)
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter"):
            sharpened = cv2.ximgproc.jointBilateralFilter(
                gray, u8, d=9, sigmaColor=75, sigmaSpace=75
            )
        else:
            sharpened = cv2.bilateralFilter(u8, d=9, sigmaColor=75, sigmaSpace=75)
        result = sharpened.astype(np.float32) / 255.0
        if result.max() > 1e-8:
            result /= result.max()
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        """Remove all registered hooks. Call after finishing all CAM generation."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        tensor: torch.Tensor,                    # (1, 3, H, W) requires_grad, on device
        class_idx: int,
        guide: Optional[np.ndarray] = None,      # (H, W, 3) uint8 for sharpening
    ) -> np.ndarray:                             # (H, W) float32 [0, 1]
        """Generate a CAM heatmap for one class.

        Parameters
        ----------
        tensor    : pre-processed input tensor with requires_grad=True
        class_idx : index of the class to explain
        guide     : original RGB image for bilateral sharpening (FPN variants)

        Returns
        -------
        Normalised heatmap in [0, 1] at the same spatial resolution as `tensor`.
        """
        H, W = tensor.shape[2], tensor.shape[3]
        self.model.zero_grad()
        self.model(tensor)[0, class_idx].backward(retain_graph=False)

        level_cams: List[np.ndarray] = []
        weights: List[float] = []

        for name, w in self.layer_cfg:
            a = self._store[name]["act"]
            g = self._store[name]["grad"]
            if a is None or g is None:
                continue
            level_cams.append(self._cam_from_acts_grads(a, g, (H, W)))
            weights.append(w)

        if not level_cams:
            return np.zeros((H, W), dtype=np.float32)

        total = sum(weights)
        fused = sum(w / total * c for w, c in zip(weights, level_cams))
        if fused.max() > 1e-8:
            fused /= fused.max()

        if self.sharpen and guide is not None:
            guide_r = cv2.resize(guide, (W, H))
            fused = self._bilateral_sharpen(fused, guide_r)

        return fused.astype(np.float32)


# ── Convenience factory ────────────────────────────────────────────────────────

def make_all_engines(model: nn.Module) -> Dict[str, CAMEngine]:
    """Instantiate all four gradient-based CAM engines on a shared model.

    Returns
    -------
    dict with keys: 'std_gradcam', 'fpn_gradcam', 'std_layercam', 'fpn_layercam'
    """
    return {
        "std_gradcam":   CAMEngine(model, STD_CFG, use_layercam=False),
        "fpn_gradcam":   CAMEngine(model, FPN_CFG, use_layercam=False),
        "std_layercam":  CAMEngine(model, STD_CFG, use_layercam=True),
        "fpn_layercam":  CAMEngine(model, FPN_CFG, use_layercam=True),
    }
