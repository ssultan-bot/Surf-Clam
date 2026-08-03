"""
Fast human-in-the-loop expansion of the training set across the FULL survey
(not just the original 38-image "Reader comparison" folder). For each image,
the model's current best-guess line is drawn already:

    ENTER (no clicks)     accept the model's line as-is -- near-zero effort
                          when the prediction already looks right
    click points          draw a corrected line instead (same as
                          label_line.py: click root->tail, in order)
    u                     undo last corrective click
    r                     clear corrective clicks (back to the model's line)
    s                     skip this image entirely (too ambiguous/damaged)
    q                     quit -- progress saved, resume later

Either way, the accepted line (yours or the model's) is saved to
data/manual_lines/<tag>.npy, exactly like label_line.py, so
lines_to_masks.py + train_unet.py pick it up automatically next run.

Images already labeled are skipped. Pass --n to control the sample size
pulled from the wider survey.

Run:
    ./.venv/bin/python ml/review_and_label.py --n 60
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from importlib import import_module
draw = import_module("02_draw_lines")
from infer import centerline_ml, segment_ml

HERE = os.path.dirname(os.path.abspath(__file__))
LINES_DIR = os.path.join(HERE, "data", "manual_lines")
os.makedirs(LINES_DIR, exist_ok=True)

SURVEY_ROOT = os.path.join(
    config.PROJECT_DIR, "Surf Clam hinge images", "NOAA Surf Clam Survey"
)
YEAR_DIRS = ["1986", "2015", "2011", "2016", "2012", "2008", "2013", "2014"]

WIN_W, WIN_H = 1400, 460


def gather_candidates():
    """All real jpgs across the year folders, tagged with a unique id that
    includes the year (filenames aren't guaranteed unique across years)."""
    out = []
    for year in YEAR_DIRS:
        d = os.path.join(SURVEY_ROOT, year)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".jpg") and not f.startswith("._"):
                tag = f"{year}__{os.path.splitext(f)[0]}"
                out.append((tag, os.path.join(d, f)))
    return out


def predicted_line(img, gray):
    """Returns (edge, is_real) -- edge is (N,2) original-resolution points
    left-to-right by x; is_real[i] is False for bridged/placeholder
    stretches where the model had no signal (see infer.bridge_gaps)."""
    edge, is_real, mask = centerline_ml(gray, bgr_for_color=img)
    if mask.sum() == 0:
        return None, None
    return edge, is_real


def review_one(img_bgr, tag, model_pts, model_is_real):
    h, w = img_bgr.shape[:2]
    frac_real = model_is_real.mean() if model_is_real is not None and len(model_is_real) else 0.0
    win_name = (f"{tag} | {frac_real*100:.0f}% real coverage | "
                f"ENTER accept | click to correct | u undo r reset s skip q quit")

    model_pts_win = None
    if model_pts is not None:
        model_pts_win = [(int(x * WIN_W / w), int(y * WIN_H / h)) for x, y in model_pts]

    clicks = []

    def render():
        canvas = cv2.resize(img_bgr, (WIN_W, WIN_H))
        if model_pts_win is not None and not clicks:
            # draw real (yellow, solid-ish) vs bridged/placeholder (red, thin)
            # segments distinctly -- a placeholder segment is a "no data
            # here" flag, not a real trace, and should look like one
            pts = np.array(model_pts_win, dtype=np.int32)
            for i in range(1, len(pts)):
                real = model_is_real is None or (model_is_real[i - 1] and model_is_real[i])
                color = (0, 200, 255) if real else (0, 0, 255)
                thickness = 2 if real else 1
                cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), color, thickness)
            cv2.putText(canvas, "YELLOW = model's guess, RED = no-data placeholder. ENTER to accept, or click to correct.",
                        (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        for i, p in enumerate(clicks):
            color = (0, 0, 255) if i == 0 else (255, 255, 0)
            cv2.circle(canvas, p, 5, color, -1)
            if i > 0:
                cv2.line(canvas, clicks[i - 1], p, (255, 255, 0), 2)
        cv2.putText(canvas, f"{len(clicks)} correction points | u undo  r reset  s skip  ENTER confirm  q quit",
                    (16, WIN_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.imshow(win_name, render())

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)
    cv2.setMouseCallback(win_name, on_mouse)
    cv2.imshow(win_name, render())

    result = "confirmed"
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:                                  # ENTER
            if clicks and len(clicks) < 2:
                continue  # need >=2 points if correcting; ignore stray single click+enter
            break
        if key in (ord("u"), ord("U")) and clicks:
            clicks.pop()
            cv2.imshow(win_name, render())
        if key in (ord("r"), ord("R")):
            clicks.clear()
            cv2.imshow(win_name, render())
        if key in (ord("s"), ord("S")):
            result = "skipped"
            break
        if key in (ord("q"), ord("Q")):
            result = "quit"
            break
    cv2.destroyAllWindows()

    if result != "confirmed":
        return result, None

    if clicks:
        pts_orig = np.array([[px * w / WIN_W, py * h / WIN_H] for px, py in clicks], dtype=float)
        return "confirmed", pts_orig
    else:
        if model_pts is None:
            return "skipped", None  # nothing to accept
        return "confirmed", model_pts


def quick_coverage(path):
    """Fast (no despike/bridging) estimate of how much of the shell the
    model is currently confident about, for ranking candidates -- full
    centerline_ml is overkill just to sort a big pool.

    Must be measured against the FULL plausible shell extent (from the
    classical segment(), same reference infer.centerline_ml uses for gap
    bridging), not the ML mask's own bounding box -- a tiny, totally wrong
    blob trivially "covers" 100% of its own bounding box, which is exactly
    the bug that made the first version of this function useless (it scored
    the known-broken 26_3 as a perfect 1.0)."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    classical_xs = np.where(draw.segment(gray).any(axis=0))[0]
    if len(classical_xs) == 0:
        return None
    x_lo, x_hi = classical_xs.min(), classical_xs.max()
    full_span = x_hi - x_lo + 1

    ml_mask = segment_ml(gray, bgr_for_color=img)
    if ml_mask.sum() == 0:
        return 0.0
    ml_xs = np.where(ml_mask.any(axis=0))[0]
    return len(ml_xs) / full_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="how many candidate images to review")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prioritize-worst", action="store_true",
                     help="pre-scan a larger pool and review the model's worst "
                          "(lowest-coverage) images first, instead of random order")
    ap.add_argument("--scan-pool", type=int, default=150,
                     help="how many candidates to pre-scan when --prioritize-worst is set")
    args = ap.parse_args()

    candidates = gather_candidates()
    already = {os.path.splitext(f)[0] for f in os.listdir(LINES_DIR) if f.endswith(".npy")}
    remaining = [(tag, path) for tag, path in candidates if tag not in already]

    random.seed(args.seed)
    random.shuffle(remaining)

    if args.prioritize_worst:
        pool = remaining[:args.scan_pool]
        print(f"pre-scanning {len(pool)} candidates for coverage...")
        scored = []
        for tag, path in pool:
            cov = quick_coverage(path)
            if cov is not None:
                scored.append((cov, tag, path))
        scored.sort(key=lambda t: t[0])  # worst (lowest coverage) first
        batch = [(tag, path) for cov, tag, path in scored[:args.n]]
        print("worst coverage in this batch:",
              [f"{tag}:{cov:.2f}" for cov, tag, path in scored[:5]])
    else:
        batch = remaining[:args.n]

    print(f"{len(candidates)} candidates total, {len(already)} already labeled, "
          f"reviewing {len(batch)} of {len(remaining)} remaining")

    for tag, path in batch:
        img = cv2.imread(path)
        if img is None:
            print(f"SKIP {tag}: unreadable")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            model_pts, model_is_real = predicted_line(img, gray)
        except Exception as e:
            print(f"SKIP {tag}: model failed ({e})")
            continue

        status, pts = review_one(img, tag, model_pts, model_is_real)
        if status == "quit":
            print("Quit early. Progress saved -- run again to resume with a new sample.")
            break
        if status == "skipped":
            print(f"skipped {tag}")
            continue

        np.save(os.path.join(LINES_DIR, tag + ".npy"), pts)
        # track original image path too, since tag alone doesn't encode the
        # year subfolder needed by lines_to_masks.py
        with open(os.path.join(LINES_DIR, tag + ".path"), "w") as fh:
            fh.write(path)
        print(f"saved {tag}: {len(pts)} points")

    n_done = len([f for f in os.listdir(LINES_DIR) if f.endswith(".npy")])
    print(f"\n{n_done} images labeled total so far.")


if __name__ == "__main__":
    main()
