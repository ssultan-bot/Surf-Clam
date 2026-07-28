"""
Same comparison as test_centerline_batch.py (old bottom-edge trace vs. new
centerline trace, run over 10 sample images), but instead of hand-picked
root coordinates, YOU click the root (hinge/umbo) for each image in a live
window -- same control as `pick_root_click` in 02_draw_lines.py itself:
click, R to reset, ENTER to confirm and move to the next image.

Run this on your own machine (needs a display):
    ./.venv/bin/python test_centerline_batch_interactive.py

Writes overlays + a summary report into intermediate/centerline_test/.
"""
import os
import sys

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

N_IMAGES = 10


def run_one(img_path, tag):
    img = cv2.imread(img_path)
    if img is None:
        return {"file": tag, "ok": False, "error": "could not read image"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = mod.segment(gray)
    if mask.sum() == 0:
        return {"file": tag, "ok": False, "error": "empty segmentation mask"}

    root_xy = mod.pick_root_click(img, config.WIN_W, config.WIN_H)

    result = {"file": tag, "ok": True, "img_shape": img.shape[:2], "root_xy": root_xy.tolist()}

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
        result[name] = {"length_px": round(length_px, 1), "n_points": len(path)}

    return result


def main():
    files = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(".jpg") and not f.startswith("._")
    )[:N_IMAGES]

    report_lines = [f"Centerline vs. bottom-edge trace test (interactive root clicks) — {len(files)} images",
                     "=" * 90]
    n_ok = 0
    for f in files:
        tag = os.path.splitext(f)[0]
        print(f"\n--- {tag} --- click the root, press ENTER ---")
        try:
            res = run_one(os.path.join(IMG_DIR, f), tag)
        except Exception as e:
            res = {"file": tag, "ok": False, "error": f"{type(e).__name__}: {e}"}

        if res["ok"]:
            n_ok += 1
            b, c = res["bottom_edge"], res["centerline"]
            report_lines.append(
                f"OK   {res['file']:32s} shape={res['img_shape']} root={res['root_xy']}\n"
                f"       bottom_edge: length={b['length_px']:>8}px\n"
                f"       centerline:  length={c['length_px']:>8}px"
            )
        else:
            report_lines.append(f"FAIL {res['file']:32s} -> {res['error']}")

    report_lines.append("=" * 90)
    report_lines.append(f"{n_ok}/{len(files)} images processed without error")

    report = "\n".join(report_lines)
    print(report)
    with open(os.path.join(OUT_DIR, "report_interactive.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
