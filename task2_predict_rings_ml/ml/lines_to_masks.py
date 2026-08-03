"""
Turn the hand-drawn lines from label_line.py into training pairs: the same
letterboxed image as auto_label.py produces, but the mask is now a thin
band around YOUR clicked centerline (spline-smoothed through the points)
instead of a whole-shell segmentation mask. This is a direct, correct
supervision signal for "where is the growth-band centerline", rather than
hoping it falls out of a shell mask.

Run after label_line.py has produced at least a few images:
    ./.venv/bin/python ml/lines_to_masks.py
"""
import os

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

HERE = os.path.dirname(os.path.abspath(__file__))
LINES_DIR = os.path.join(HERE, "data", "manual_lines")
IMAGES_OUT = os.path.join(HERE, "data", "images")
MASKS_OUT = os.path.join(HERE, "data", "masks")
os.makedirs(IMAGES_OUT, exist_ok=True)
os.makedirs(MASKS_OUT, exist_ok=True)

import sys
sys.path.insert(0, os.path.dirname(HERE))
import config

IMG_DIR = os.path.join(
    config.PROJECT_DIR,
    "Surf Clam hinge images",
    "NOAA Surf Clam Survey",
    "Reader comparison selected images",
)

TRAIN_SIZE = 512
BAND_HALF_WIDTH_FRAC = 0.02   # half-width of the drawn band, as a fraction of
                              # the longer image side -- matches roughly how
                              # thick the growth band actually looks


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


def band_mask(shape, points, half_width):
    """Spline-smooth the clicked points, then draw a thick polyline -- a
    band mask centered on the true growth-band centerline."""
    h, w = shape
    pts = np.asarray(points, dtype=float)
    if len(pts) >= 4:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts) - 1))
        xs, ys = splev(np.linspace(0, 1, 300), tck)
    else:
        xs, ys = pts[:, 0], pts[:, 1]

    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(mask, [poly], False, 255, thickness=max(1, int(round(half_width * 2))))
    return mask


def main():
    if not os.path.isdir(LINES_DIR) or not os.listdir(LINES_DIR):
        print(f"No manual lines found in {LINES_DIR} -- run label_line.py first.")
        return

    tags = sorted(os.path.splitext(f)[0] for f in os.listdir(LINES_DIR) if f.endswith(".npy"))
    print(f"{len(tags)} manually-labeled images found")

    kept = 0
    for tag in tags:
        # images labeled via review_and_label.py (broader survey, not just
        # the "Reader comparison" folder) carry a sidecar .path file with
        # their real location, since the tag alone doesn't encode the year
        # subfolder.
        sidecar = os.path.join(LINES_DIR, tag + ".path")
        if os.path.exists(sidecar):
            with open(sidecar) as fh:
                img_path = fh.read().strip()
        else:
            img_path = os.path.join(IMG_DIR, tag + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"SKIP {tag}: image not found at {img_path}")
            continue

        pts = np.load(os.path.join(LINES_DIR, tag + ".npy"))
        h, w = img.shape[:2]
        half_width = BAND_HALF_WIDTH_FRAC * max(h, w)
        mask = band_mask((h, w), pts, half_width)

        img_sq = letterbox_square(img, TRAIN_SIZE)
        mask_sq = letterbox_square(mask, TRAIN_SIZE, is_mask=True)

        cv2.imwrite(os.path.join(IMAGES_OUT, f"{tag}.png"), img_sq)
        cv2.imwrite(os.path.join(MASKS_OUT, f"{tag}.png"), mask_sq)
        print(f"OK   {tag}: {len(pts)} points -> band mask")
        kept += 1

    print(f"\n{kept} training pairs written to {IMAGES_OUT} / {MASKS_OUT}")
    print("Now run: ./.venv/bin/python ml/train_unet.py")


if __name__ == "__main__":
    main()
