"""
Hand-draw the true growth-band centerline for a set of images. This gives
the model a direct, correct signal for what it should predict, instead of
inferring it indirectly from a whole-shell segmentation mask (which is what
auto_label.py did, and which the model then reproduces the flaws of).

Controls per image:
    left-click   add a point, in order from the ROOT (hinge/umbo) to the
                 TAIL end -- just enough points to define the curve, usually
                 6-10 is plenty, doesn't need to be pixel-perfect
    u            undo last point
    r            reset (clear all points for this image)
    ENTER        confirm this image (needs >= 2 points) and move to the next
    s            skip this image (e.g. too damaged/ambiguous to trace)
    q            quit early -- keeps everything labeled so far

Points are saved as original-image pixel coordinates (Nx2 float arrays) to
ml/data/manual_lines/<tag>.npy, one file per image, so you can stop and
resume across sessions (already-labeled images are skipped automatically).

Run:
    ./.venv/bin/python ml/label_line.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

IMG_DIR = os.path.join(
    config.PROJECT_DIR,
    "Surf Clam hinge images",
    "NOAA Surf Clam Survey",
    "Reader comparison selected images",
)

HERE = os.path.dirname(os.path.abspath(__file__))
LINES_DIR = os.path.join(HERE, "data", "manual_lines")
os.makedirs(LINES_DIR, exist_ok=True)

WIN_W, WIN_H = 1400, 460


def label_one(img_bgr, tag):
    h, w = img_bgr.shape[:2]
    win_name = f"{tag}  |  click ROOT->TAIL points | u undo | r reset | s skip | ENTER confirm | q quit"
    pts = []

    def render():
        canvas = cv2.resize(img_bgr, (WIN_W, WIN_H))
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i == 0 else (255, 255, 0)
            cv2.circle(canvas, p, 5, color, -1)
            if i > 0:
                cv2.line(canvas, pts[i - 1], p, (255, 255, 0), 2)
        cv2.putText(canvas, f"{len(pts)} points  |  root->tail  |  u undo  r reset  s skip  ENTER confirm  q quit",
                    (16, WIN_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.imshow(win_name, render())

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)
    cv2.setMouseCallback(win_name, on_mouse)
    cv2.imshow(win_name, render())

    result = "confirmed"
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(pts) >= 2:          # ENTER
            break
        if key in (ord("u"), ord("U")) and pts:  # undo
            pts.pop()
            cv2.imshow(win_name, render())
        if key in (ord("r"), ord("R")):          # reset
            pts.clear()
            cv2.imshow(win_name, render())
        if key in (ord("s"), ord("S")):          # skip
            result = "skipped"
            break
        if key in (ord("q"), ord("Q")):          # quit
            result = "quit"
            break
    cv2.destroyAllWindows()

    if result != "confirmed":
        return result, None

    pts_orig = np.array([[px * w / WIN_W, py * h / WIN_H] for px, py in pts], dtype=float)
    return "confirmed", pts_orig


def main():
    files = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(".jpg") and not f.startswith("._")
    )

    remaining = [f for f in files
                 if not os.path.exists(os.path.join(LINES_DIR, os.path.splitext(f)[0] + ".npy"))]
    print(f"{len(files)} total images, {len(files) - len(remaining)} already labeled, "
          f"{len(remaining)} left")

    for f in remaining:
        tag = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(IMG_DIR, f))
        if img is None:
            print(f"SKIP {tag}: unreadable")
            continue

        status, pts = label_one(img, tag)
        if status == "quit":
            print("Quit early. Progress saved -- run again to resume.")
            break
        if status == "skipped":
            print(f"skipped {tag}")
            continue

        np.save(os.path.join(LINES_DIR, tag + ".npy"), pts)
        print(f"saved {tag}: {len(pts)} points")

    n_done = len([f for f in os.listdir(LINES_DIR) if f.endswith(".npy")])
    print(f"\n{n_done}/{len(files)} images labeled so far.")


if __name__ == "__main__":
    main()
