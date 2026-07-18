"""
monusac_utils.py
----------------
Data loading utilities for MoNuSAC 2020.

Key fix applied here: the class name in MoNuSAC XML files is stored under
    <Attributes><Attribute Name="Epithelial">
and NOT in the <Annotation Name=""> attribute (which is always empty).
This fix is essential — without it, all annotations are ignored.

cv2.fillPoly fix: polygon arrays must be wrapped in np.ascontiguousarray
before being passed to cv2.fillPoly to avoid a crash on non-contiguous
memory layouts from polygon vertex extraction.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# Class name → channel index mapping
XML_CLASS_MAP: Dict[str, int] = {
    "Epithelial": 0, "epithelial": 0, "EPITHELIAL": 0,
    "Lymphocyte": 1, "lymphocyte": 1, "LYMPHOCYTE": 1,
    "Macrophage": 2, "macrophage": 2, "MACROPHAGE": 2,
    "Neutrophil": 3, "neutrophil": 3, "NEUTROPHIL": 3,
    "Ambiguous":  -1, "ambiguous": -1,
}

CLASS_NAMES: List[str] = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]
NUM_CLASSES = 4
IMG_SIZE    = 256


def parse_monusac_xml(xml_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """Parse a MoNuSAC ImageScope XML annotation file.

    Parameters
    ----------
    xml_path : path to the .xml annotation file
    img_h    : image height in pixels (for mask array shape)
    img_w    : image width in pixels

    Returns
    -------
    (img_h, img_w, 4) uint8 binary mask — one channel per class

    Notes
    -----
    Class identification uses the Attributes block, not the Annotation Name
    attribute (which is always empty in MoNuSAC XMLs). This is the critical
    bug fix relative to naive ImageScope XML parsers.
    """
    mask = np.zeros((img_h, img_w, NUM_CLASSES), dtype=np.uint8)
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        return mask

    for ann in root.findall("Annotation"):
        class_idx = -1

        # Primary: look in <Attributes><Attribute Name="...">
        attrs_block = ann.find("Attributes")
        if attrs_block is not None:
            first_attr = attrs_block.find("Attribute")
            if first_attr is not None:
                raw       = first_attr.get("Name", "").strip()
                class_idx = XML_CLASS_MAP.get(raw,
                            XML_CLASS_MAP.get(raw.capitalize(), -1))

        # Fallback: try the Annotation Name / PartOfGroup attribute
        if class_idx < 0:
            raw       = ann.get("Name", ann.get("PartOfGroup", "")).strip()
            class_idx = XML_CLASS_MAP.get(raw,
                        XML_CLASS_MAP.get(raw.capitalize(), -1))

        if class_idx < 0:
            continue  # Ambiguous or unrecognised class

        for region in ann.findall(".//Region"):
            vertices = region.findall(".//Vertex")
            if len(vertices) < 3:
                continue

            pts = []
            for v in vertices:
                x = v.get("X") or v.get("x")
                y = v.get("Y") or v.get("y")
                if x is not None and y is not None:
                    pts.append([float(x), float(y)])

            if len(pts) < 3:
                continue

            pts_arr       = np.array(pts, dtype=np.float32)
            pts_arr[:, 0] = np.clip(pts_arr[:, 0], 0, img_w - 1)
            pts_arr[:, 1] = np.clip(pts_arr[:, 1], 0, img_h - 1)
            pts_int       = pts_arr.astype(np.int32).reshape((-1, 1, 2))

            # ascontiguousarray required — cv2.fillPoly crashes on non-contiguous arrays
            channel = np.ascontiguousarray(mask[..., class_idx])
            cv2.fillPoly(channel, [pts_int], color=1)
            mask[..., class_idx] = channel

    return mask


def load_monusac_sample(
    tif_path: Path,
    xml_path: Path,
    target_size: int = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one MoNuSAC image and its annotation mask.

    Parameters
    ----------
    tif_path    : path to the .tif image
    xml_path    : path to the corresponding .xml annotation
    target_size : resize both image and mask to this square size

    Returns
    -------
    (img_resized, mask_resized)
        img_resized  : (target_size, target_size, 3) uint8 RGB
        mask_resized : (target_size, target_size, 4) float32 binary
    """
    img_pil  = Image.open(tif_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    mask_orig = parse_monusac_xml(xml_path, orig_h, orig_w).astype(np.float32)

    img_resized = np.array(
        img_pil.resize((target_size, target_size), Image.BILINEAR), dtype=np.uint8
    )

    if orig_h != target_size or orig_w != target_size:
        mask_resized = np.stack([
            cv2.resize(mask_orig[..., c], (target_size, target_size),
                       interpolation=cv2.INTER_NEAREST)
            for c in range(NUM_CLASSES)
        ], axis=-1)
    else:
        mask_resized = mask_orig

    return img_resized, mask_resized.astype(np.float32)


def build_monusac_index(root: Path, verbose: bool = True) -> pd.DataFrame:
    """Scan a MoNuSAC data directory and build a sample index DataFrame.

    Parameters
    ----------
    root    : root directory containing patient subdirectories
    verbose : print summary statistics

    Returns
    -------
    DataFrame with columns:
        patient_id, sample_name, tif_path, xml_path, is_test,
        orig_h, orig_w
    """
    records = []
    for pdir in sorted(d for d in root.iterdir() if d.is_dir()):
        pid = pdir.name
        for tif in sorted(pdir.glob("*.tif")):
            xml = tif.with_suffix(".xml")
            if not xml.exists():
                continue
            with Image.open(tif) as im:
                orig_w, orig_h = im.size
            records.append({
                "patient_id":  pid,
                "sample_name": tif.stem,
                "tif_path":    str(tif),
                "xml_path":    str(xml),
                "is_test":     False,  # set externally after split
                "orig_h":      orig_h,
                "orig_w":      orig_w,
            })

    df = pd.DataFrame(records)
    if verbose:
        print(f"Found {len(df)} samples from {df['patient_id'].nunique()} patients")
        print(f"Native size: H {df['orig_h'].min()}–{df['orig_h'].max()}, "
              f"W {df['orig_w'].min()}–{df['orig_w'].max()}")
    return df


def load_all_samples(
    df: pd.DataFrame,
    target_size: int = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all images and masks from a MoNuSAC index DataFrame.

    Parameters
    ----------
    df          : output of build_monusac_index()
    target_size : resize target

    Returns
    -------
    (images, masks)
        images : (N, target_size, target_size, 3) uint8
        masks  : (N, target_size, target_size, 4) float32
    """
    N      = len(df)
    images = np.zeros((N, target_size, target_size, 3), dtype=np.uint8)
    masks  = np.zeros((N, target_size, target_size, NUM_CLASSES), dtype=np.float32)

    for i, row in enumerate(tqdm(df.itertuples(), total=N, desc="Loading MoNuSAC")):
        img, msk    = load_monusac_sample(
            Path(row.tif_path), Path(row.xml_path), target_size)
        images[i]   = img
        masks[i]    = msk

    return images, masks


def patient_level_split(
    df: pd.DataFrame,
    test_frac: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply a random patient-level train/test split.

    All images from a given patient are assigned to the same partition,
    preventing patient-level data leakage between train and test sets.

    Parameters
    ----------
    df        : index DataFrame from build_monusac_index()
    test_frac : fraction of patients to assign to the test set
    seed      : RNG seed for reproducibility

    Returns
    -------
    df with 'is_test' column set
    """
    rng          = np.random.default_rng(seed)
    all_patients = df["patient_id"].unique().copy()
    rng.shuffle(all_patients)
    n_test  = max(1, int(len(all_patients) * test_frac))
    test_pt = set(all_patients[:n_test])
    df      = df.copy()
    df["is_test"] = df["patient_id"].isin(test_pt)
    print(f"Patient-level split (seed={seed}):")
    print(f"  Test:  {len(test_pt)} patients → {df['is_test'].sum()} images")
    print(f"  Train: {len(all_patients)-len(test_pt)} patients → {(~df['is_test']).sum()} images")
    return df
