"""
scorecam_engine.py
------------------
Gradient-free Score-CAM implementation with corrected baseline subtraction.

Score-CAM [Wang et al., CVPR Workshops 2020] generates channel importance
weights by measuring each channel's causal contribution to the class score
via masked forward passes, rather than backpropagating gradients.

CRITICAL: Baseline Subtraction
-------------------------------
The importance score for channel k must be BASELINE-SUBTRACTED:

    s_k = σ(f_c(X * M_k)) - σ(f_c(X))

Without baseline subtraction, all scores are non-negative and cluster
near the model's existing class confidence, causing the weighted sum to
collapse to a spatially uniform heatmap (flat after normalisation) that
binarises to full image coverage, producing IoU ≈ 0.

This bug appears in many public Score-CAM implementations. The corrected
engine in this file produces valid, spatially meaningful heatmaps.

Usage
-----
    from src.scorecam_engine import ScoreCAMEngine

    engine = ScoreCAMEngine(model, layer_path='backbone.features.denseblock4')
    cam    = engine.generate(img_tensor, class_idx=0)
    engine.remove_hooks()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCORECAM_LAYER = "backbone.features.denseblock4"
SCORECAM_BATCH = 64    # channels per GPU batch; reduce to 32 if OOM on 8 GB VRAM


class ScoreCAMEngine:
    """Gradient-free Score-CAM for DenseNet169 multi-label classification.

    Parameters
    ----------
    model      : trained DenseNet169MultiLabel in eval mode
    layer_path : dotted attribute path to the target convolutional layer.
                 Default targets denseblock4 (1,664 channels, 8×8 spatial
                 at 256×256 input).
    batch_size : number of masked forward passes per GPU batch.
                 64 is safe on 8 GB VRAM; reduce if OOM.

    Notes
    -----
    Score-CAM requires K forward passes per class per image, where K is
    the number of channels in the target layer (K=1,664 for denseblock4).
    For 500 PanNuke test images × ~2 active classes, this is ~1.66M passes.
    Expected runtime: 1–2 GPU hours.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_path: str = SCORECAM_LAYER,
        batch_size: int = SCORECAM_BATCH,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._acts: Optional[torch.Tensor] = None

        # Navigate to target layer and register one forward hook
        layer = model
        for k in layer_path.split("."):
            layer = getattr(layer, k)

        self._handle = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach())
        )

    def remove_hooks(self) -> None:
        """Remove hook. Call after all processing to avoid hook accumulation."""
        self._handle.remove()

    @torch.no_grad()
    def generate(
        self,
        img_tensor: torch.Tensor,  # (1, 3, H, W) normalised, on DEVICE
        class_idx: int,
    ) -> np.ndarray:               # (H, W) float32 in [0, 1]
        """Generate a Score-CAM heatmap for one class.

        Algorithm
        ---------
        1. Baseline forward pass → activation maps A^k + baseline score σ(f_c(X))
        2. Upsample each A^k to input resolution; min-max normalise → mask M_k ∈ [0,1]
        3. For each channel k: forward pass on (X * M_k) → s_k = σ(f_c(X*M_k)) − σ(f_c(X))
        4. CAM = ReLU(Σ_k s_k · A^k), upsampled to input resolution

        Parameters
        ----------
        img_tensor : (1, 3, H, W) normalised tensor on DEVICE.
                     No requires_grad needed — Score-CAM is gradient-free.
        class_idx  : index of the class to explain.

        Returns
        -------
        Normalised heatmap in [0, 1] at the same spatial resolution as img_tensor.
        """
        self.model.eval()
        H, W = img_tensor.shape[2], img_tensor.shape[3]
        device = img_tensor.device

        # ── Step 1: baseline forward pass ─────────────────────────────────────
        # Single pass captures both activation maps and the baseline class score.
        # Clone acts immediately — later masked passes will overwrite self._acts.
        logits_base = self.model(img_tensor)
        score_base  = torch.sigmoid(logits_base[0, class_idx]).item()
        acts        = self._acts.squeeze(0).clone()   # (K, h, w)
        K, h, w     = acts.shape

        # ── Step 2: upsample + per-channel min-max normalise ──────────────────
        acts_up = F.interpolate(
            acts.unsqueeze(0),              # (1, K, h, w)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                        # (K, H, W)

        a_min = acts_up.flatten(1).min(dim=1).values.view(K, 1, 1)
        a_max = acts_up.flatten(1).max(dim=1).values.view(K, 1, 1)
        masks = ((acts_up - a_min) / (a_max - a_min).clamp(min=1e-8)).clamp(0.0, 1.0)

        # ── Step 3: masked forward passes — baseline-subtracted scores ─────────
        # Positive s_k: channel k activates class c above the unmasked baseline.
        # Negative s_k: channel k suppresses class c.
        # Without subtraction, all s_k ≈ score_base > 0, collapsing the sum.
        scores = torch.zeros(K, device=device)
        for start in range(0, K, self.batch_size):
            end        = min(start + self.batch_size, K)
            batch_k    = end - start
            img_batch  = img_tensor.expand(batch_k, -1, -1, -1)
            mask_batch = masks[start:end].unsqueeze(1)   # (B, 1, H, W)
            logits     = self.model(img_batch * mask_batch)
            scores[start:end] = torch.sigmoid(logits[:, class_idx]) - score_base

        # ── Step 4: signed weighted sum on the cloned baseline activations ─────
        w_k     = scores.view(K, 1, 1)                   # (K, 1, 1)
        cam_raw = F.relu((w_k * acts).sum(dim=0))         # (h, w)

        cam_up = F.interpolate(
            cam_raw.unsqueeze(0).unsqueeze(0),            # (1, 1, h, w)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()                          # (H, W)

        cam_up = np.maximum(cam_up, 0.0)
        if cam_up.max() > 1e-8:
            cam_up /= cam_up.max()
        return cam_up.astype(np.float32)
