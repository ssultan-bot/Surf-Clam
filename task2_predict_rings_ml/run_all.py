# run_all.py
# ----------------------------------------------------------------------
# THE WHOLE TASK 2 PIPELINE in one run.
#
# Scripts 01-06 teach each step and show a comparison chart along the way.
# This file does everything start to finish, in one go, and only shows the
# FINAL result. Use it once you understand the steps.
#
#   1. FastSAM finds shapes -> you click the shell mask.
#   2. You draw line(s) along the growth direction.
#   3. Read gray brightness along each line.
#   4. CLAHE boosts contrast.
#   5. Gaussian smooths the brightness curve.
#   6. Find the dark-band valleys (= rings), show the result, save CSVs.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# (First run downloads the FastSAM model; needs internet once.)
# ----------------------------------------------------------------------

import os                                            # file/folder toolbox
import config                                        # our settings

# Cache folders must be set BEFORE importing the AI and plotting libraries.
os.environ.setdefault("YOLO_CONFIG_DIR", config.YOLO_CACHE)
os.environ.setdefault("MPLCONFIGDIR", config.MPL_CACHE)

import cv2                                           # images, windows, mouse, CLAHE
import numpy as np                                   # math on arrays
import pandas as pd                                  # save CSV tables
import matplotlib                                    # plotting (core)
matplotlib.use("TkAgg")                              # open plots in a normal pop-up window
import matplotlib.pyplot as plt                      # the drawing tools
from scipy.ndimage import gaussian_filter1d          # smooths a line of numbers
from scipy.interpolate import interp1d               # fills in points between samples
from scipy.signal import find_peaks                  # finds peaks (used to find valleys)
from ultralytics import FastSAM                      # the AI segmentation model


def pick_mask(masks_resized, img_rgb):
    """Show all masks in a grid; let the user click one; return its index."""
    n = len(masks_resized)
    ROWS = (n + config.COLS - 1) // config.COLS      # rows needed in the grid
    selected = [None]                                # remembers the clicked mask

    def build(highlight=None):                       # draw the grid of thumbnails
        canvas = np.zeros((ROWS * config.PH, config.COLS * config.PW, 3), dtype=np.uint8)
        for i, mask in enumerate(masks_resized):
            r, c = divmod(i, config.COLS)
            overlay = img_rgb.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 180, 255]) * 0.5).astype(np.uint8)
            thumb = cv2.cvtColor(cv2.resize(overlay, (config.PW, config.PH)), cv2.COLOR_RGB2BGR)
            color = (0, 255, 100) if highlight == i else (0, 255, 255)
            cv2.putText(thumb, f"Mask {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if highlight == i:
                cv2.rectangle(thumb, (3, 3), (config.PW - 3, config.PH - 3), (0, 255, 100), 4)
            canvas[r * config.PH:(r + 1) * config.PH, c * config.PW:(c + 1) * config.PW] = thumb
        return canvas

    def on_click(event, x, y, flags, param):         # react to a left-click
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        idx = (y // config.PH) * config.COLS + (x // config.PW)
        if 0 <= idx < n:
            selected[0] = idx
            print(f"Selected mask: {idx}")
            cv2.imshow("Step1: Click mask, press ENTER", build(idx))

    cv2.namedWindow("Step1: Click mask, press ENTER", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Step1: Click mask, press ENTER", config.COLS * config.PW, ROWS * config.PH)
    cv2.setMouseCallback("Step1: Click mask, press ENTER", on_click)
    cv2.imshow("Step1: Click mask, press ENTER", build())
    while True:
        if cv2.waitKey(20) == 13 and selected[0] is not None:   # ENTER once a mask is picked
            break
    cv2.destroyAllWindows()
    return selected[0]


def draw_lines(bg_display):
    """Let the user draw lines on the masked photo; return them (window coords)."""
    COLORS = [(0, 255, 100), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
    lines, cur_line, is_drawing = [], [], [False]

    def render():
        canvas = bg_display.copy()
        for i, line in enumerate(lines):
            c = COLORS[i % len(COLORS)]
            for j in range(1, len(line)):
                cv2.line(canvas, line[j - 1], line[j], c, 2)
            cv2.putText(canvas, f"L{i + 1}", line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        for j in range(1, len(cur_line)):
            cv2.line(canvas, cur_line[j - 1], cur_line[j], (255, 255, 255), 2)
        cv2.putText(canvas, f"Lines:{len(lines)}  LBtn=draw  RBtn=undo  ENTER=confirm",
                    (10, config.WIN_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return canvas

    def on_draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing[0] = True
            cur_line.clear()
            cur_line.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and is_drawing[0]:
            cur_line.append((x, y))
            cv2.imshow("Step2: Draw lines, ENTER to confirm", render())
        elif event == cv2.EVENT_LBUTTONUP and is_drawing[0]:
            is_drawing[0] = False
            cur_line.append((x, y))
            if len(cur_line) > 2:
                lines.append(list(cur_line))
                print(f"Line {len(lines)} recorded ({len(cur_line)} pts)")
            cur_line.clear()
            cv2.imshow("Step2: Draw lines, ENTER to confirm", render())
        elif event == cv2.EVENT_RBUTTONDOWN:
            if lines:
                lines.pop()
                print(f"Undo - {len(lines)} lines remaining")
                cv2.imshow("Step2: Draw lines, ENTER to confirm", render())

    cv2.namedWindow("Step2: Draw lines, ENTER to confirm", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Step2: Draw lines, ENTER to confirm", config.WIN_W, config.WIN_H)
    cv2.setMouseCallback("Step2: Draw lines, ENTER to confirm", on_draw)
    cv2.imshow("Step2: Draw lines, ENTER to confirm", render())
    while True:
        if cv2.waitKey(20) == 13 and len(lines) > 0:    # ENTER once at least one line exists
            break
    cv2.destroyAllWindows()
    return lines


def process_line(win_pts, w, h):
    """Turn one drawn line into a clean, smooth left-to-right pixel path."""
    orig = [(int(px * w / config.WIN_W), int(py * h / config.WIN_H)) for px, py in win_pts]
    xs = np.array([p[0] for p in orig])
    ys = np.array([p[1] for p in orig])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xs_u, inv = np.unique(xs, return_inverse=True)
    ys_u = np.array([ys[inv == i].mean() for i in range(len(xs_u))])
    if len(xs_u) < 2:
        return xs_u, ys_u.astype(int)
    x_full = np.arange(xs_u[0], xs_u[-1] + 1)
    y_full = interp1d(xs_u, ys_u, kind="linear")(x_full)
    y_smooth = gaussian_filter1d(y_full, sigma=config.SMOOTH_SIGMA)
    return x_full, y_smooth.astype(int)


# ----------------------------------------------------------------------
# MAIN program.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    img = cv2.imread(config.IMG_PATH)                 # read the photo
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # 1) FastSAM segmentation.
    print("Running FastSAM... (first time downloads the model)")
    model = FastSAM(config.MODEL)
    results = model(config.IMG_PATH, device=config.DEVICE, retina_masks=True,
                    imgsz=config.IMGSZ, conf=config.CONF, iou=config.IOU, save=False, verbose=False)
    masks = results[0].masks.data.cpu().numpy()
    masks_resized = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
    print(f"Total masks: {len(masks_resized)}")
    if len(masks_resized) == 0:
        raise ValueError("FastSAM found no masks. Lower CONF in config.py.")

    # 2) Pick the shell mask.
    sel = pick_mask(masks_resized, img_rgb)
    best_mask = masks_resized[sel]

    # 3) Draw lines on the masked photo.
    overlay_bg = img_rgb.copy()
    overlay_bg[best_mask > 0] = (overlay_bg[best_mask > 0] * 0.55 + np.array([0, 180, 255]) * 0.45).astype(np.uint8)
    bg_display = cv2.resize(cv2.cvtColor(overlay_bg, cv2.COLOR_RGB2BGR), (config.WIN_W, config.WIN_H))
    lines = draw_lines(bg_display)
    print(f"Total lines: {len(lines)}")

    # 4) CLAHE on the gray image (contrast boost).
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=(config.CLAHE_TILE, config.CLAHE_TILE))
    gray_clahe_img = clahe.apply(img_gray)

    # 5) For each line: path -> gray -> CLAHE -> smooth -> find valleys.
    all_data = []
    for i, line in enumerate(lines):
        x_arr, y_arr = process_line(line, w, h)
        y_arr = np.clip(y_arr, 0, h - 1)
        x_arr = np.clip(x_arr, 0, w - 1)
        gray_raw = np.array([img_gray[y_arr[j], x_arr[j]] for j in range(len(x_arr))], dtype=float)
        gray_clahe = np.array([gray_clahe_img[y_arr[j], x_arr[j]] for j in range(len(x_arr))], dtype=float)
        gray_smooth = gaussian_filter1d(gray_clahe, sigma=config.GRAY_SIGMA)
        gray_range = gray_smooth.max() - gray_smooth.min()
        valleys, _ = find_peaks(-gray_smooth, distance=config.MIN_DIST,
                                prominence=gray_range * config.PROMINENCE)
        all_data.append({"idx": i + 1, "x": x_arr, "y": y_arr,
                         "gray_raw": gray_raw, "gray_clahe": gray_clahe, "gray_smooth": gray_smooth,
                         "valleys": valleys, "valley_x": x_arr[valleys], "valley_y": y_arr[valleys],
                         "valley_gray": gray_smooth[valleys]})
        print(f"Line {i + 1}: {len(valleys)} rings detected")

    # 6) Show the final result and save CSVs.
    COLORS = ["cyan", "orange", "magenta", "lime", "yellow"]
    n = len(all_data)
    fig, axes = plt.subplots(n + 1, 1, figsize=(18, 5 * (n + 1)))
    if n + 1 == 1:
        axes = [axes]
    img_masked = img_rgb.copy()
    img_masked[best_mask == 0] = 0
    axes[0].imshow(img_masked)
    for i, d in enumerate(all_data):
        c = COLORS[i % len(COLORS)]
        axes[0].plot(d["x"], d["y"], color=c, linewidth=2, label=f"Line {d['idx']} - {len(d['valleys'])} rings")
        axes[0].scatter(d["valley_x"], d["valley_y"], color="red", s=50, zorder=5)
        for vx in d["valley_x"]:
            axes[0].axvline(x=vx, color="red", linewidth=0.8, alpha=0.45)
    axes[0].set_title("Detected growth rings (red markers = dark-band valleys)")
    axes[0].legend(loc="upper right")
    axes[0].axis("off")
    for i, d in enumerate(all_data):
        c = COLORS[i % len(COLORS)]
        ax = axes[i + 1]
        ax.plot(d["x"], d["gray_smooth"], color=c, linewidth=2.0, label="smoothed brightness")
        ax.scatter(d["valley_x"], d["valley_gray"], color="red", s=80, zorder=5, label=f"{len(d['valleys'])} rings")
        for vx in d["valley_x"]:
            ax.axvline(x=vx, color="red", linewidth=0.8, alpha=0.4)
        ax.set_title(f"Line {d['idx']}: {len(d['valleys'])} rings")
        ax.set_xlabel("x (pixel column)")
        ax.set_ylabel("gray value (0-255)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(d["x"][0], d["x"][-1])
    plt.tight_layout()
    print("\nA window opened with the final result. Close it to save the CSVs.")
    plt.show()

    base = os.path.splitext(config.IMG_PATH)[0]
    for d in all_data:
        pd.DataFrame({"x": d["x"], "y": d["y"],
                      "gray_raw": d["gray_raw"].astype(int),
                      "gray_clahe": d["gray_clahe"].astype(int),
                      "gray_smooth": d["gray_smooth"].round(2)}).to_csv(f"{base}_line{d['idx']}_pixels.csv", index=False)
        pd.DataFrame({"ring_no": np.arange(1, len(d["valley_x"]) + 1),
                      "x": d["valley_x"], "y": d["valley_y"],
                      "gray": d["valley_gray"].round(2)}).to_csv(f"{base}_line{d['idx']}_rings.csv", index=False)
        print(f"Saved line {d['idx']}: pixels CSV + rings CSV ({len(d['valley_x'])} rings)")
    print("Done.")
