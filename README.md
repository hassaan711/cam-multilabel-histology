# Understanding CAM Behaviour Under Multi-Label Contamination in Histological Nuclei Classification

**A Cross-Dataset Analysis of Gradient Treatment, Resolution Strategy, Multi-Label Contamination, and Gradient-Free Attribution Using DenseNet169 on PanNuke and MoNuSAC 2020**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This repository contains the full experimental code for a mechanistic study of Class Activation Mapping (CAM) behaviour under multi-label co-occurrence in histological nuclei classification.

**Key findings:**
- Score-CAM (gradient-free, with corrected baseline subtraction) outperforms all gradient-based methods on both datasets
- All four gradient-based methods show perfect monotonic IoU degradation with label cardinality (Spearman ρ = −1.000)
- The FPN pyramid main effect reverses on MoNuSAC, a resolution ablation confirms this is caused by variable-size patch resampling, not architecture
- LayerCAM amplifies multi-label contamination relative to GradCAM (Wilcoxon p = 0.009), establishing a precision–robustness trade-off
- LayerCAM retains 3.7× more spatial signal than GradCAM on misclassified images

---

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── model.py              # DenseNet169MultiLabel classifier
│   ├── cam_engine.py         # Gradient-based CAMEngine (all 4 methods)
│   ├── scorecam_engine.py    # Gradient-free ScoreCAMEngine (corrected)
│   ├── evaluation.py         # IoU, bootstrap CI, contamination analysis
│   ├── preprocessing.py      # Transforms, inference, overlay utilities
│   └── monusac_utils.py      # MoNuSAC XML parser and data loading
│
├── notebooks/
│   ├── pannuke_densenet169.ipynb           # PanNuke training
│   ├── pannuke_layercam_comparison.ipynb   # 2×2 factorial CAM (PanNuke)
│   ├── pannuke_scorecam.ipynb              # Score-CAM (PanNuke)
│   ├── pannuke_monusac_cam.ipynb           # 2×2 factorial CAM (MoNuSAC)
│   ├── monusac_scorecam.ipynb              # Score-CAM (MoNuSAC)
│   └── monusac_resolution_ablation.ipynb   # Resolution ablation
│
├── configs/
│   ├── pannuke.yaml           # PanNuke hyperparameters and paths
│   └── monusac.yaml           # MoNuSAC hyperparameters and paths
│
├── requirements.txt
└── README.md
```

---

## Datasets

### PanNuke
- **Source:** https://jgamper.github.io/PanNukeDataset/
- **Size:** 7,702 patches (256×256 px), 5 nuclear classes, 3 folds
- **Classes:** Neoplastic, Non-neoplastic Epithelial, Inflammatory, Connective, Dead
- **Used:** All 7,702 images. 500 held-out test images (seed=42, ~167/fold)

### MoNuSAC 2020
- **Source:** https://monusac-2020.grand-challenge.org/
- **Size:** 209 patches (variable size → resized to 256×256), 4 nuclear classes, 47 patients
- **Classes:** Epithelial, Lymphocyte, Macrophage, Neutrophil
- **Used:** All 209 images. Patient-level 80/20 split (seed=42) → 36 test images

> **Important (MoNuSAC):** The class name in MoNuSAC XML files is stored under
> `<Attributes><Attribute Name="Epithelial">`, **not** in `<Annotation Name="">`.
> The parser in `src/monusac_utils.py` implements this fix — naive parsers will
> silently drop all annotations.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cam-multilabel-histology.git
cd cam-multilabel-histology
pip install -r requirements.txt
```

GPU with ≥ 8 GB VRAM recommended. All experiments were run on a single GPU.

---

## Quick Start

### 1. Training (PanNuke)

Open and run `notebooks/pannuke_densenet169.ipynb`.

Key settings in Cell 4:
```python
N_TEST_HOLD = 500   # images held out for CAM evaluation
SEED        = 42
NUM_EPOCHS  = 30
```

This saves `outputs/best_densenet169_pannuke.pth` and `outputs/test_predictions.json`.

### 2. Training (MoNuSAC)

Open and run `notebooks/pannuke_monusac_cam.ipynb` through the training section (Cells 1–24).

### 3. CAM Evaluation

Run the following notebooks in order after training:

| Notebook | Dataset | Content |
|---|---|---|
| `pannuke_layercam_comparison.ipynb` | PanNuke | 2×2 factorial GradCAM/LayerCAM/FPN |
| `pannuke_scorecam.ipynb` | PanNuke | Score-CAM + comparison |
| `pannuke_monusac_cam.ipynb` | MoNuSAC | 2×2 factorial GradCAM/LayerCAM/FPN |
| `monusac_scorecam.ipynb` | MoNuSAC | Score-CAM + comparison |
| `monusac_resolution_ablation.ipynb` | MoNuSAC | Native vs 256×256 resolution |

### 4. Using the `src` modules directly

```python
import torch
from src.model import load_checkpoint
from src.cam_engine import make_all_engines, FPN_CFG
from src.scorecam_engine import ScoreCAMEngine
from src.preprocessing import preprocess, predict
from src.evaluation import compute_iou, bootstrap_macro_iou_ci

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = load_checkpoint('outputs/best_densenet169_pannuke.pth',
                        num_classes=5, device=device)

# Gradient-based CAMs
engines = make_all_engines(model)
img_t   = preprocess(img_rgb, requires_grad=True, device=device)
cam     = engines['fpn_layercam'].generate(img_t, class_idx=0, guide=img_rgb)
for eng in engines.values():
    eng.remove_hooks()

# Score-CAM
sc_engine = ScoreCAMEngine(model, batch_size=64)
img_t     = preprocess(img_rgb, requires_grad=False, device=device)
cam_sc    = sc_engine.generate(img_t, class_idx=0)
sc_engine.remove_hooks()

# Evaluate
iou = compute_iou(cam, gt_mask_channel, thr=0.50)
```

---

## Important Implementation Notes

### Score-CAM Baseline Subtraction (Critical)

The standard Score-CAM formula computes channel importance as:

```
s_k = sigmoid(f_c(X * M_k)) - sigmoid(f_c(X))   ← CORRECT
```

Many public implementations omit the `- sigmoid(f_c(X))` term. Without it,
all scores are non-negative and cluster near the model's existing class
confidence. The weighted sum collapses to a spatially uniform heatmap that
binarises to full image coverage, producing IoU ≈ 0.

The corrected implementation is in `src/scorecam_engine.py`.

### MoNuSAC Test Split

The test split uses a **forced random patient-level 80/20 split** (seed=42),
regardless of any official patient list. This ensures all notebooks use
identical test patients. The split is applied unconditionally — do not wrap
it in a conditional check.

### CAMEngine Hook Lifecycle

Always call `engine.remove_hooks()` after all CAM generation for an engine.
Failing to do so accumulates hooks across loop iterations, causing gradient
tensor accumulation and eventually OOM.

---

## Reproducing Paper Results

All experiments use:
- `SEED = 42` for the primary run
- Seeds `[0, 1, 42]` for the multi-seed stability analysis
- `N_TEST_HOLD = 500` for PanNuke (set in training notebook)
- All 36 MoNuSAC test images (no subsampling cap)

Expected results (macro IoU@0.50):

| Method | PanNuke (n=500) | MoNuSAC (n=36) |
|---|---|---|
| Standard GradCAM | 0.0880 | 0.0824 |
| FPN-GradCAM | 0.0789 | 0.0630 |
| Standard LayerCAM | 0.0872 | 0.0849 |
| FPN-LayerCAM | 0.1118 | 0.0896 |
| **Score-CAM** | **0.1326** | **0.1083** |

Classification stability (Macro AUC, mean ± std across 3 seeds):
- PanNuke: 0.929 ± 0.006
- MoNuSAC: 0.880 ± 0.011

---

## Citation

If you use this code, please cite:

```bibtex
@article{cam_multilabel_histology_2026,
  title   = {Understanding CAM Behaviour Under Multi-Label contamination in
             Histological Nuclei Classification},
  journal = {Medical Image Analysis},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Both datasets (PanNuke and MoNuSAC 2020) are derived from TCGA open-access
data and are subject to their respective dataset licenses.
