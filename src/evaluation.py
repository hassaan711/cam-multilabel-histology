"""
evaluation.py
-------------
Evaluation utilities:
    - IoU between binarised CAM and GT nucleus mask
    - Bootstrap 95% CI for macro IoU
    - IoU vs label cardinality (Spearman ρ)
    - Pairwise multi-label interference matrix
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_iou(
    cam: np.ndarray,    # (H, W) float32 [0, 1]
    gt_ch: np.ndarray,  # (H, W) binary or float
    thr: float = 0.50,
) -> float:
    """IoU between a binarised CAM and a GT mask channel.

    Returns NaN when the GT channel has no positive pixels
    (class absent in this image), so it can be safely excluded
    from per-class means without biasing the statistic.
    """
    gt_bin = gt_ch > 0
    if not gt_bin.any():
        return float("nan")
    pred  = cam >= thr
    inter = (pred & gt_bin).sum()
    union = (pred | gt_bin).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_macro_iou_ci(
    df: pd.DataFrame,
    method_key: str,
    class_keys: List[str],
    threshold: float = 0.50,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for macro IoU of one method at one threshold.

    Resamples images (rows) with replacement N=n_boot times.
    Each resample computes per-class mean IoU → nanmean across classes.

    Parameters
    ----------
    df          : DataFrame where each row is one test image
    method_key  : method key (e.g. 'std_gradcam')
    class_keys  : list of class name strings matching column name fragments
    threshold   : IoU binarisation threshold
    n_boot      : number of bootstrap resamples
    ci          : confidence level (0.95 → 95%)
    seed        : RNG seed for reproducibility

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)

    Notes
    -----
    Uses percentile bootstrap (no bias correction). Appropriate for n ≥ 30.
    For n < 30 (e.g. MoNuSAC with 36 test images) interpret intervals with
    caution due to limited sample size.
    """
    rng   = np.random.default_rng(seed)
    thr_s = f"{threshold:.2f}"

    # Collect per-image IoU arrays for each class
    col_arrays = []
    for ck in class_keys:
        col = f"{method_key}_iou_{thr_s}_{ck}"
        arr = df[col].values.astype(float) if col in df.columns else np.full(len(df), np.nan)
        col_arrays.append(arr)

    iou_mat  = np.column_stack(col_arrays)   # (N, C)
    n_images = len(iou_mat)

    # Point estimate
    per_class_means = np.nanmean(iou_mat, axis=0)
    point_est       = float(np.nanmean(per_class_means))

    # Bootstrap distribution
    boot_macros = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx             = rng.integers(0, n_images, size=n_images)
        resample        = iou_mat[idx]
        cls_means       = np.nanmean(resample, axis=0)
        boot_macros[b]  = np.nanmean(cls_means)

    alpha = 1.0 - ci
    lo    = float(np.nanpercentile(boot_macros, 100 * alpha / 2))
    hi    = float(np.nanpercentile(boot_macros, 100 * (1 - alpha / 2)))

    return point_est, lo, hi


# ── Cardinality analysis ───────────────────────────────────────────────────────

def iou_by_cardinality(
    df: pd.DataFrame,
    method_key: str,
    class_keys: List[str],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Mean IoU stratified by label cardinality (number of co-occurring classes).

    Parameters
    ----------
    df         : DataFrame with 'cardinality' column and IoU columns
    method_key : method key string
    class_keys : list of class name strings
    threshold  : IoU threshold

    Returns
    -------
    DataFrame with columns: cardinality, mean, sem, n
    """
    thr_s = f"{threshold:.2f}"
    rows  = []
    for card in sorted(df["cardinality"].unique()):
        sub  = df[df["cardinality"] == card]
        cols = [f"{method_key}_iou_{thr_s}_{ck}" for ck in class_keys
                if f"{method_key}_iou_{thr_s}_{ck}" in df.columns]
        vals = np.concatenate([sub[c].dropna().values for c in cols]) if cols else np.array([])
        if len(vals) == 0:
            continue
        rows.append({
            "cardinality": card,
            "mean": float(np.nanmean(vals)),
            "sem":  float(np.nanstd(vals) / max(np.sqrt(len(vals)), 1)),
            "n":    len(vals),
        })
    return pd.DataFrame(rows)


def spearman_cardinality(
    df_card: pd.DataFrame,
) -> Tuple[float, float]:
    """Spearman ρ between cardinality and mean IoU.

    Parameters
    ----------
    df_card : output of iou_by_cardinality()

    Returns
    -------
    (rho, p_value)
    """
    if len(df_card) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(df_card["cardinality"], df_card["mean"])
    return float(rho), float(p)


# ── Pairwise interference ──────────────────────────────────────────────────────

def pairwise_interference(
    df: pd.DataFrame,
    method_key: str,
    class_names: List[str],
    threshold: float = 0.50,
    min_n: int = 5,
) -> pd.DataFrame:
    """Compute pairwise multi-label interference matrix.

    For each class pair (c, c'):
        ΔIoU = mean_IoU(class_c | class_c' absent)
               − mean_IoU(class_c | class_c' present)

    A positive ΔIoU indicates that the presence of class c' degrades
    the localisation of class c (interference).

    Parameters
    ----------
    df          : DataFrame with gt_{class}, pred_{class}, and IoU columns
    method_key  : method key string
    class_names : list of class name strings
    threshold   : IoU threshold
    min_n       : minimum number of samples required in each condition

    Returns
    -------
    DataFrame with columns:
        class_c, class_cp, delta_iou, n_absent, n_present, p_val, significant
    """
    thr_s  = f"{threshold:.2f}"
    rows   = []
    n_cls  = len(class_names)

    for ci in range(n_cls):
        cname    = class_names[ci]
        iou_col  = f"{method_key}_iou_{thr_s}_{cname}"
        if iou_col not in df.columns:
            continue

        # Images where class c is correctly predicted present
        base = df[
            (df[f"gt_{cname}"] == 1) & (df[f"pred_{cname}"] == 1)
        ][[iou_col] + [f"gt_{class_names[cp]}" for cp in range(n_cls)]].dropna()

        for cp in range(n_cls):
            if ci == cp:
                continue
            cname_p  = class_names[cp]
            absent   = base[base[f"gt_{cname_p}"] == 0][iou_col].values
            present  = base[base[f"gt_{cname_p}"] == 1][iou_col].values

            if len(absent) < min_n or len(present) < min_n:
                continue

            delta   = float(np.nanmean(absent)) - float(np.nanmean(present))
            _, pval = stats.ttest_ind(absent, present, equal_var=False, nan_policy="omit")

            rows.append({
                "class_c"    : cname,
                "class_cp"   : cname_p,
                "delta_iou"  : round(delta, 4),
                "n_absent"   : len(absent),
                "n_present"  : len(present),
                "p_val"      : round(float(pval), 4),
                "significant": bool(pval < 0.05),
            })

    return pd.DataFrame(rows)


# ── Factorial decomposition ────────────────────────────────────────────────────

def factorial_decomposition(
    sg: float,  # Standard GradCAM macro IoU
    fg: float,  # FPN-GradCAM macro IoU
    sl: float,  # Standard LayerCAM macro IoU
    fl: float,  # FPN-LayerCAM macro IoU
) -> Dict[str, float]:
    """2×2 factorial decomposition of IoU effects.

    Returns
    -------
    dict with keys:
        layercam_main  : LayerCAM main effect
        pyramid_main   : Pyramid main effect
        interaction    : LayerCAM × Pyramid interaction term
    """
    return {
        "layercam_main": round(((sl - sg) + (fl - fg)) / 2, 4),
        "pyramid_main":  round(((fg - sg) + (fl - sl)) / 2, 4),
        "interaction":   round((fl - fg) - (sl - sg), 4),
    }
