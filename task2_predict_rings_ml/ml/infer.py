"""
Drop-in replacement for 02_draw_lines.segment() using the trained U-Net
instead of Otsu thresholding. Same contract: takes a grayscale image,
returns a filled binary mask (uint8, 0/255) at the ORIGINAL resolution.

Usage from another script:
    from ml.infer import segment_ml
    mask = segment_ml(gray, bgr_for_color=img)   # img optional, improves accuracy
"""
import os
import sys

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `from ml.infer import ...` callers still resolve model.py
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from model import SmallUNet
from importlib import import_module
_draw = import_module("02_draw_lines")
CKPT_PATH = os.path.join(HERE, "shell_unet.pt")
SIZE = 512

_model = None


def _load_model():
    global _model
    if _model is None:
        m = SmallUNet()
        m.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        m.eval()
        _model = m
    return _model


def _letterbox_square(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, scale, y0, x0


def predict_prob(gray, bgr_for_color=None):
    """Returns the model's raw per-pixel confidence (0-1 float), resized back
    to the ORIGINAL image resolution -- no thresholding. Use this (not
    segment_ml's binary mask) when you need sub-pixel-accurate centerline
    position rather than just "is this shell or not"."""
    model = _load_model()
    h, w = gray.shape[:2]

    if bgr_for_color is not None:
        rgb = cv2.cvtColor(bgr_for_color, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    sq, scale, y0, x0 = _letterbox_square(rgb, SIZE)
    x = torch.from_numpy(sq).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].numpy()

    nh, nw = int(round(h * scale)), int(round(w * scale))
    probs_cropped = probs[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(probs_cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def segment_ml(gray, bgr_for_color=None, thresh=0.5):
    """gray: grayscale image (H, W). bgr_for_color: optional color image
    (H, W, 3) for a better RGB input to the model; falls back to a gray
    triplicated to 3 channels if not given."""
    probs = predict_prob(gray, bgr_for_color)
    mask = (probs > thresh).astype(np.uint8)

    # keep the largest connected component, fill holes -- same cleanup as
    # the classical segment(), the model output can have small speckles
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = 1 + int(np.argmax(areas))
        mask = np.uint8(lab == big)

    return ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255


def centerline_from_prob(prob, mask):
    """Confidence-weighted centroid per column, restricted to the column's
    masked extent. Far more robust to a lopsided/over-thick predicted blob
    than taking the naive (top+bottom)/2 of a hard-thresholded mask: pixels
    near the blob's fuzzy, low-confidence edges barely move the average,
    so the line tracks where the model is actually MOST confident, not
    wherever the 0.5 cutoff happened to land."""
    xs = np.where(mask.any(axis=0))[0]
    mids = np.empty(len(xs), dtype=float)
    ys_full = np.arange(mask.shape[0], dtype=float)
    for i, x in enumerate(xs):
        col_mask = mask[:, x] > 0
        w = prob[:, x] * col_mask
        wsum = w.sum()
        if wsum < 1e-6:
            idx = np.where(col_mask)[0]
            mids[i] = (idx.min() + idx.max()) / 2.0
        else:
            mids[i] = (ys_full * w).sum() / wsum
    return np.column_stack([xs.astype(float), mids])


def bridge_gaps(edge, x_lo, x_hi, max_native_gap=3):
    """Fixes the "model predicts nothing over part of the shell" failure
    mode: rather than just leaving the line missing there, bridge across it.

    - Internal gaps (a stretch between two columns the model DID predict,
      more than max_native_gap apart): linearly interpolate between the
      confident points on either side -- we know where the curve starts
      and ends, we just don't have model output in between.
    - Boundary gaps (the model's confident region doesn't reach x_lo/x_hi,
      the plausible full extent of the shell from a coarser reference mask):
      hold the nearest confident y flat out to the boundary. This is a
      conservative placeholder, not a claimed measurement -- there's no
      reasonable way to guess the curve's shape with zero model signal.

    Returns (points, is_real) -- is_real[i] is False for interpolated/
    placeholder points, so callers can draw them distinctly (e.g. dashed)
    instead of passing off a flat guess as a real trace. A large fraction of
    False means "the model mostly failed on this image", which callers
    should surface, not hide.
    """
    edge = edge[np.argsort(edge[:, 0])]
    xs, ys = edge[:, 0], edge[:, 1]

    out_x = [xs[0]]
    out_y = [ys[0]]
    out_real = [True]
    for i in range(1, len(xs)):
        gap = xs[i] - xs[i - 1]
        if gap > max_native_gap:
            n_fill = int(gap) - 1
            fill_x = np.linspace(xs[i - 1], xs[i], n_fill + 2)[1:-1]
            fill_y = np.linspace(ys[i - 1], ys[i], n_fill + 2)[1:-1]
            out_x.extend(fill_x)
            out_y.extend(fill_y)
            out_real.extend([False] * len(fill_x))
        out_x.append(xs[i])
        out_y.append(ys[i])
        out_real.append(True)
    xs, ys = np.array(out_x), np.array(out_y)
    is_real = np.array(out_real, dtype=bool)

    if xs[0] > x_lo + max_native_gap:
        lead_x = np.arange(x_lo, xs[0])
        xs = np.concatenate([lead_x, xs])
        ys = np.concatenate([np.full(len(lead_x), ys[0]), ys])
        is_real = np.concatenate([np.zeros(len(lead_x), dtype=bool), is_real])
    if xs[-1] < x_hi - max_native_gap:
        trail_x = np.arange(xs[-1] + 1, x_hi + 1)
        xs = np.concatenate([xs, trail_x])
        ys = np.concatenate([ys, np.full(len(trail_x), ys[-1])])
        is_real = np.concatenate([is_real, np.zeros(len(trail_x), dtype=bool)])

    return np.column_stack([xs, ys]), is_real


def centerline_ml(gray, bgr_for_color=None):
    """Full ML centerline pipeline for one image: predict, extract a
    confidence-weighted centerline, despike, and bridge any gaps against
    the classical segment()'s rough shell extent (used ONLY to know how far
    left/right the shell plausibly goes -- its actual y-values are not
    trusted, that's the model's job). Returns (edge, is_real, ml_mask):
    edge is an (N,2) array of original-resolution points ascending by x;
    is_real[i] is False where the model had no signal and the point is a
    bridged placeholder, not a real prediction."""
    ml_mask = segment_ml(gray, bgr_for_color=bgr_for_color)
    if ml_mask.sum() == 0:
        # total model failure on this image -- fall back to classical extent
        # entirely so the caller still gets *something* rather than nothing
        classical_mask = _draw.segment(gray)
        edge = _draw.bottom_edge_path(classical_mask)
        return edge, np.zeros(len(edge), dtype=bool), ml_mask

    prob = predict_prob(gray, bgr_for_color=bgr_for_color)
    edge = centerline_from_prob(prob, ml_mask)
    edge = _draw.despike(edge)

    classical_mask = _draw.segment(gray)
    classical_xs = np.where(classical_mask.any(axis=0))[0]
    x_lo = min(edge[:, 0].min(), classical_xs.min()) if len(classical_xs) else edge[:, 0].min()
    x_hi = max(edge[:, 0].max(), classical_xs.max()) if len(classical_xs) else edge[:, 0].max()

    edge, is_real = bridge_gaps(edge, x_lo, x_hi)
    return edge, is_real, ml_mask
