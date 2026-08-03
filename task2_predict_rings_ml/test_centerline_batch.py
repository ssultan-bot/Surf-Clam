"""
Headless comparison of the old bottom-edge trace vs. the new centerline trace,
run over 10 sample images. `pick_root_click` needs a live window; for this
batch test the root (hinge/umbo) point for each image was instead picked by
eye once, from full-resolution gridded crops, and is hardcoded below in
original-image pixel coordinates -- interactive use is unaffected and still
uses a live click.

Writes overlays + a summary report into intermediate/centerline_test/.
"""
import os
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config
from importlib import import_module

mod = import_module("02_draw_lines")

IMG_DIR = os.path.join(
    config.PROJECT_DIR,
    "Surf Clam hinge images",
    "NOAA Surf Clam Survey",
    "Reader comparison selected images",
)

OUT_DIR = os.path.join(config.INTERMEDIATE_DIR, "centerline_test")
os.makedirs(OUT_DIR, exist_ok=True)

# tag -> (root_x, root_y) in full original-image pixel coordinates, read directly
# off gridded 900px-wide crops of each image's hinge end (not estimated from a
# downscaled whole-image preview, which was too lossy to click precisely on a
# 4000+px-wide image and was the main source of bad anchors in the first pass).
HAND_PICKED_ROOTS = {
    "Surf_Clam_198604_Summer_23_1":  (3900, 610),
    "Surf_Clam_198604_Summer_29_29": (2700, 700),
    "Surf_Clam_198604_Summer_30_15": (2750, 150),
    "Surf_Clam_198604_Summer_31_10": (3460, 190),
    "Surf_Clam_198604_Summer_31_22": (2150, 280),
    "Surf_Clam_198604_Summer_37_22": (2200, 220),
    "Surf_Clam_198604_Summer_6_3":   (5, 500),
    "Surf_Clam_1986_Summer_255_2":   (3630, 290),
    "Surf_Clam_201160_Summer_42_1":  (20, 60),
    "Surf_Clam_201160_Summer_43_1":  (10, 110),
}


def hand_root(tag, img_w, img_h):
    return np.array(HAND_PICKED_ROOTS[tag], dtype=float)


def run_one(img_path, tag):
    img = cv2.imread(img_path)
    if img is None:
        return {"file": tag, "ok": False, "error": "could not read image"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = mod.segment_best_available(gray, bgr_for_color=img)
    if mask.sum() == 0:
        return {"file": tag, "ok": False, "error": "empty segmentation mask"}

    h, w = img.shape[:2]
    root_xy = hand_root(tag, w, h)

    result = {"file": tag, "ok": True, "img_shape": img.shape[:2]}

    for name, path_fn in (("bottom_edge", mod.bottom_edge_path),
                           ("centerline", mod.centerline_path)):
        edge = path_fn(mask)
        edge = mod.despike(edge)
        edge = mod.trim_near(edge, root_xy)
        edge = mod.relax_near_root(edge, root_xy, gray)
        sigma = config.CENTERLINE_SMOOTH_SIGMA if name == "centerline" else 18
        path = mod.smooth(edge, sigma=sigma)
        path = mod.orient_root_first(path, root_xy)
        path = mod.anchor_to_root(path, root_xy)

        overlay_img = mod.overlay(img, path)
        out_path = os.path.join(OUT_DIR, f"{tag}__{name}.png")
        cv2.imwrite(out_path, overlay_img)

        length_px = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
        gap_to_root = float(np.hypot(*(path[0] - root_xy)))
        result[name] = {
            "length_px": round(length_px, 1),
            "n_points": len(path),
            "root_gap_px": round(gap_to_root, 1),
        }

    return result


def main():
    files = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(".jpg") and not f.startswith("._")
        and os.path.splitext(f)[0] in HAND_PICKED_ROOTS
    )

    report_lines = [f"Centerline vs. bottom-edge trace test — {len(files)} images", "=" * 90]
    n_ok = 0
    for f in files:
        tag = os.path.splitext(f)[0]
        try:
            res = run_one(os.path.join(IMG_DIR, f), tag)
        except Exception:
            res = {"file": tag, "ok": False, "error": "exception (see traceback below)"}
            report_lines.append(f"[TRACE] {tag}\n{traceback.format_exc()}")

        if res["ok"]:
            n_ok += 1
            b, c = res["bottom_edge"], res["centerline"]
            report_lines.append(
                f"OK   {res['file']:32s} shape={res['img_shape']}\n"
                f"       bottom_edge: length={b['length_px']:>8}px  root_gap={b['root_gap_px']:>6}px\n"
                f"       centerline:  length={c['length_px']:>8}px  root_gap={c['root_gap_px']:>6}px"
            )
        else:
            report_lines.append(f"FAIL {res['file']:32s} -> {res['error']}")

    report_lines.append("=" * 90)
    report_lines.append(f"{n_ok}/{len(files)} images processed without error")
    report_lines.append(
        "root_gap_px = distance between the hand-picked hinge point and where the "
        "traced line actually starts (small = line starts at the root)."
    )

    report = "\n".join(report_lines)
    print(report)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
