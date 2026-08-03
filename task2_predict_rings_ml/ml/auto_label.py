"""
Build a training set with zero manual labeling.

For every source image: run FastSAM (already used interactively in
01_segment_pick.py) to get several candidate object masks, then pick
whichever candidate best overlaps the existing classical segment() mask
from 02_draw_lines.py. FastSAM contributes accurate object boundaries
(it understands "this is one solid object"); the classical mask
contributes rough location (which object is the shell). Combining them
gives a decent label automatically, no clicking or painting needed.

Images/masks are saved downscaled to TRAIN_SIZE for training speed.

Run:
    ./.venv/bin/python ml/auto_label.py
"""
import os
import sys

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo_cache")

import cv2
import numpy as np
from scipy import ndimage as ndi
from ultralytics import FastSAM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from importlib import import_module
draw = import_module("02_draw_lines")

IMG_DIR = os.path.join(
    config.PROJECT_DIR,
    "Surf Clam hinge images",
    "NOAA Surf Clam Survey",
    "Reader comparison selected images",
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_OUT = os.path.join(OUT_DIR, "data", "images")
MASKS_OUT = os.path.join(OUT_DIR, "data", "masks")
os.makedirs(IMAGES_OUT, exist_ok=True)
os.makedirs(MASKS_OUT, exist_ok=True)

TRAIN_SIZE = 512          # square training resolution
MIN_IOU = 0.15            # below this, FastSAM likely found the wrong object -> skip
MIN_MASK_FRAC = 0.15      # candidate covering less than 15% of the image is probably a
                          # fragment (e.g. just the debris chunk next to the shell), not
                          # the whole shell -- genuine full-shell masks run ~0.3-0.5
MAX_FILL_RATIO = 0.65     # candidate filling more of its own bounding box than this is
                          # probably shell+fused-debris, not a clean elongated shell -> reject
                          # (found by comparing known-good vs known-bad candidates: clean
                          # shells run ~0.4-0.6, debris-fused blobs run ~0.75+)
MIN_LARGEST_COMPONENT_FRAC = 0.8   # candidate must be ~one solid blob, not speckled


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def fill_ratio(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 1.0
    bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    return mask.sum() / bbox_area


def largest_component_fraction(mask):
    """A clean shell mask is one solid blob. A noisy candidate (speckled
    debris scattered across the background, or bleeding onto an unrelated
    panel) has its area split across many small pieces instead."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return areas.max() / areas.sum()


def best_fastsam_mask(fastsam_masks, ref_mask):
    best, best_score = None, -1.0
    for m in fastsam_masks:
        frac = m.mean()
        if frac < MIN_MASK_FRAC:
            continue
        if fill_ratio(m) > MAX_FILL_RATIO:
            continue
        if largest_component_fraction(m) < MIN_LARGEST_COMPONENT_FRAC:
            continue
        score = iou(m > 0, ref_mask > 0)
        if score > best_score:
            best, best_score = m, score
    return best, best_score


def letterbox_square(img, size, is_mask=False):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main():
    files = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(".jpg") and not f.startswith("._")
    )
    print(f"{len(files)} source images found")

    model = FastSAM(os.path.join(os.path.dirname(OUT_DIR), "FastSAM-s.pt"))

    kept, skipped = 0, 0
    for f in files:
        tag = os.path.splitext(f)[0]
        path = os.path.join(IMG_DIR, f)
        img = cv2.imread(path)
        if img is None:
            print(f"SKIP {tag}: unreadable")
            skipped += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        ref_mask = draw.segment(gray) > 0

        results = model(path, device="cpu", retina_masks=True, imgsz=512,
                         conf=0.3, iou=0.9, save=False, verbose=False)
        if results[0].masks is None:
            print(f"SKIP {tag}: FastSAM found nothing")
            skipped += 1
            continue

        raw_masks = results[0].masks.data.cpu().numpy()
        cands = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                 for m in raw_masks]

        mask, score = best_fastsam_mask(cands, ref_mask)
        if mask is None or score < MIN_IOU:
            print(f"SKIP {tag}: best IoU {score:.2f} < {MIN_IOU} (no confident match)")
            skipped += 1
            continue

        # final cleanup: keep only the largest blob, fill any interior holes
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (lab == big).astype(np.uint8)
        mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)

        img_sq = letterbox_square(img, TRAIN_SIZE)
        mask_sq = letterbox_square((mask * 255).astype(np.uint8), TRAIN_SIZE, is_mask=True)

        cv2.imwrite(os.path.join(IMAGES_OUT, f"{tag}.png"), img_sq)
        cv2.imwrite(os.path.join(MASKS_OUT, f"{tag}.png"), mask_sq)
        print(f"OK   {tag}: IoU {score:.2f}")
        kept += 1

    print(f"\n{kept} labeled, {skipped} skipped (out of {len(files)})")


if __name__ == "__main__":
    main()
