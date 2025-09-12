import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

def _ensure_uint8_mask(mask: np.ndarray | None, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Ensure mask is uint8 {0,1} and same HxW as image."""
    if mask is None:
        return None
    m = mask
    if m.ndim == 3:  # if color
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    m = (m > 127).astype(np.uint8)
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
    return m

def compute_fullness_cv(
    bgr: np.ndarray,
    shelf_mask: np.ndarray | None = None,
    min_blob_frac: float = 0.002,
    ar_min: float = 1.8,
) -> tuple[float, List[Dict]]:
    """
    Heuristic gap detector:
    - Low texture (Laplacian) -> candidate gaps
    - Morph close horizontally to unify spans
    - Filter small / non-horizontal blobs
    Returns:
      fullness_cv in [0,1], list of gaps with bbox/area/score
    """
    assert bgr is not None and bgr.size > 0, "Empty image"
    h, w = bgr.shape[:2]

    # Apply mask if provided
    shelf_mask = _ensure_uint8_mask(shelf_mask, (h, w))
    roi = bgr.copy()
    if shelf_mask is not None:
        roi[~shelf_mask.astype(bool)] = 0

    # 1) texture map
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tex = cv2.convertScaleAbs(np.abs(lap))
    tex = cv2.GaussianBlur(tex, (5, 5), 0)

    # 2) threshold: low-texture -> gap candidate
    thr = max(10, int(np.mean(tex) * 0.7))
    _, gap = cv2.threshold(tex, thr, 255, cv2.THRESH_BINARY_INV)

    # 3) morphology to merge horizontal spans
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    gap = cv2.morphologyEx(gap, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4) filter blobs
    cnts, _ = cv2.findContours(gap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(h * w * min_blob_frac)
    gaps: List[Dict] = []
    gap_area = 0
    for c in cnts:
        x, y, wc, hc = cv2.boundingRect(c)
        area = wc * hc
        if area < min_area:
            continue
        ar = wc / max(hc, 1)
        if ar < ar_min:
            continue
        gaps.append(
            {
                "bbox": [int(x), int(y), int(wc), int(hc)],
                "gap_area": int(area),
                "score": float(min(1.0, area / (w * h))),
            }
        )
        gap_area += area

    shelf_area = (h * w) if shelf_mask is None else int(np.count_nonzero(shelf_mask))
    shelf_area = max(shelf_area, 1)
    fullness_cv = float(max(0.0, min(1.0, 1.0 - gap_area / shelf_area)))
    return fullness_cv, gaps

def draw_gaps(bgr: np.ndarray, gaps: List[Dict]) -> np.ndarray:
    """Utility for debugging: draw green boxes for detected gaps."""
    vis = bgr.copy()
    for g in gaps:
        x, y, w, h = g["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis
