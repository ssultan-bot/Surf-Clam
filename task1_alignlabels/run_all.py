# run_all.py
# ----------------------------------------------------------------------
# THE REAL THING, all in one run.
#
# Scripts 01-05 each teach one step on their own (and 04/05 use a made-up
# example curve so they can run alone). THIS file instead does the whole
# real job from start to finish, using your REAL Excel measurements and the
# REAL curve you draw with the mouse:
#
#   1. Read the photo.
#   2. Read the real growth increments for this photo from Excel.
#   3. You draw the hinge curve with the mouse (tail -> root).
#   4. Smooth the curve and place each real measurement onto it.
#   5. Show the final two-panel picture.
#
# Nothing is saved to disk. Everything happens in memory in this one run.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# ----------------------------------------------------------------------

import os                                            # file-name helpers
import cv2                                           # read photo, drawing window, mouse
import numpy as np                                   # math on arrays
import pandas as pd                                  # read the Excel table
import matplotlib                                    # plotting library (core)
import matplotlib
matplotlib.use("MacOSX")
# show plots in a normal pop-up window (avoids a PyCharm backend bug)
import matplotlib.pyplot as plt                      # the drawing tools
from scipy.interpolate import interp1d               # "fills in" points along the curve
from scipy.ndimage import gaussian_filter1d          # smooths the curve
import config                                        # our settings


def read_increments(excel_path, file_col, image_name):
    """Read this image's real growth increments from Excel (root -> tail order)."""
    df = pd.read_excel(excel_path)                    # load the sheet as a table
    if file_col not in df.columns:                    # make sure the name column exists
        raise ValueError(f"Cannot find column '{file_col}'. Available: {list(df.columns)}")
    matched = df[df[file_col].astype(str) == image_name]               # exact name match
    if len(matched) == 0:                                              # else loose match
        matched = df[df[file_col].astype(str).str.contains(image_name, regex=False, na=False)]
    if len(matched) == 0:                                              # still nothing -> stop
        raise ValueError(f"Cannot find label row for image: {image_name}")
    row = matched.iloc[0]                                              # the first matching row
    i_cols = [c for c in df.columns if str(c).startswith("I")]         # columns I1, I2, ...
    i_series = row[i_cols].dropna()                                    # values, blanks removed
    increments = pd.to_numeric(i_series, errors="coerce").dropna().values.astype(float)  # to numbers
    if len(increments) == 0:                                          # no usable numbers -> stop
        raise ValueError("No valid increment labels found.")
    return increments                                                 # root -> tail


def draw_curve_on_image(img_bgr):
    """Open a window, let the user draw a curve, return points in IMAGE coordinates."""
    h, w = img_bgr.shape[:2]                          # real image size (for scaling)
    drawing = [False]                                 # is the mouse button held down?
    pts_win = []                                       # points in WINDOW pixels

    def render():                                      # build the picture to show
        canvas = cv2.resize(img_bgr.copy(), (config.WIN_W, config.WIN_H))   # fit to window
        if len(pts_win) > 1:                                                # connect with lines
            for i in range(1, len(pts_win)):
                cv2.line(canvas, pts_win[i - 1], pts_win[i], (0, 255, 255), 2)
        if len(pts_win) > 0:                                               # mark start and end
            cv2.circle(canvas, pts_win[0], 7, (0, 255, 0), -1)             # green = TAIL start
            cv2.putText(canvas, "TAIL START", (pts_win[0][0] + 8, pts_win[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(canvas, pts_win[-1], 7, (0, 0, 255), -1)            # red = ROOT end
            cv2.putText(canvas, "ROOT END", (pts_win[-1][0] + 8, pts_win[-1][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(canvas, "Draw TAIL -> ROOT | hold left mouse | R reset | ENTER confirm",
                    (20, config.WIN_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        return canvas

    def on_mouse(event, x, y, flags, param):           # runs on every mouse action
        if event == cv2.EVENT_LBUTTONDOWN:             # press -> start fresh
            drawing[0] = True
            pts_win.clear()
            pts_win.append((x, y))
            cv2.imshow("Draw curve from TAIL to ROOT", render())
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:   # drag -> add points
            pts_win.append((x, y))
            cv2.imshow("Draw curve from TAIL to ROOT", render())
        elif event == cv2.EVENT_LBUTTONUP and drawing[0]:   # release -> finish stroke
            drawing[0] = False
            pts_win.append((x, y))
            cv2.imshow("Draw curve from TAIL to ROOT", render())

    cv2.namedWindow("Draw curve from TAIL to ROOT", cv2.WINDOW_NORMAL)      # make the window
    cv2.resizeWindow("Draw curve from TAIL to ROOT", config.WIN_W, config.WIN_H)
    cv2.setMouseCallback("Draw curve from TAIL to ROOT", on_mouse)          # connect mouse handler
    cv2.imshow("Draw curve from TAIL to ROOT", render())                    # show starting picture
    while True:                                        # keep window alive, watch keyboard
        key = cv2.waitKey(20)
        if key == 13 and len(pts_win) >= 5:            # ENTER (and >= 5 points) -> done
            break
        if key in [ord("r"), ord("R")]:               # R -> reset
            pts_win.clear()
            cv2.imshow("Draw curve from TAIL to ROOT", render())
    cv2.destroyAllWindows()                            # close the window

    # Convert WINDOW coordinates back to REAL IMAGE coordinates.
    pts = np.array([[px * w / config.WIN_W, py * h / config.WIN_H] for px, py in pts_win], dtype=float)
    if len(pts) < 5:                                   # safety check
        raise ValueError("Too few valid curve points.")
    return pts


def smooth_and_resample(curve_pts, sigma):
    """Smooth the curve and return the clean curve + distance->x / distance->y helpers."""
    diff = np.diff(curve_pts, axis=0)                                  # drop duplicate points
    keep = np.r_[True, np.linalg.norm(diff, axis=1) > 1e-6]
    curve_pts = curve_pts[keep]
    seg = np.linalg.norm(np.diff(curve_pts, axis=0), axis=1)          # distance along the curve
    s_raw = np.r_[0, np.cumsum(seg)]
    raw_len = float(s_raw[-1])
    unique_s, idx = np.unique(s_raw, return_index=True)               # keep unique distances
    cu = curve_pts[idx]
    fx = interp1d(unique_s, cu[:, 0], kind="linear", bounds_error=False, fill_value="extrapolate")
    fy = interp1d(unique_s, cu[:, 1], kind="linear", bounds_error=False, fill_value="extrapolate")
    s_dense = np.linspace(0, raw_len, int(np.floor(raw_len)) + 1)     # one point per pixel
    x_s = gaussian_filter1d(fx(s_dense), sigma=sigma)                 # smooth x
    y_s = gaussian_filter1d(fy(s_dense), sigma=sigma)                 # smooth y
    seg2 = np.linalg.norm(np.diff(np.column_stack([x_s, y_s]), axis=0), axis=1)
    s2 = np.r_[0, np.cumsum(seg2)]
    path_len = float(s2[-1])
    fxp = interp1d(s2, x_s, kind="linear", bounds_error=False, fill_value="extrapolate")
    fyp = interp1d(s2, y_s, kind="linear", bounds_error=False, fill_value="extrapolate")
    su = np.linspace(0, path_len, int(np.floor(path_len)) + 1)        # final even sampling
    return fxp(su), fyp(su), path_len, fxp, fyp


# ----------------------------------------------------------------------
# MAIN program.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    img_bgr = cv2.imread(config.IMG_PATH)             # 1) read the photo
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_name = os.path.basename(config.IMG_PATH)

    # 2) Read the REAL increments and turn them into distances from the tail.
    increments_root_to_tail = read_increments(config.LABEL_EXCEL, config.FILE_COL, img_name)
    increments_tail_to_root = increments_root_to_tail[::-1]            # reverse to tail -> root
    cum_dist = np.cumsum(increments_tail_to_root)                     # distance from tail per ring
    total_label_len = float(cum_dist[-1])

    print(f"Image: {img_name}")
    print(f"Number of real increments: {len(increments_tail_to_root)}")
    print(f"Total real label length: {total_label_len:.1f} pixels")
    print("\nNow draw the hinge curve from TAIL to ROOT, then press ENTER...")

    # 3) You draw the real curve.
    curve_pts = draw_curve_on_image(img_bgr)

    # 4) Smooth the curve, then project the REAL increments onto it.
    x_path, y_path, path_len, fx_path, fy_path = smooth_and_resample(curve_pts, config.SMOOTH_PATH_SIGMA)
    if config.FIT_LABELS_TO_DRAWN_CURVE:              # optional: stretch labels to fill the curve
        s_all = cum_dist / total_label_len * path_len
    else:                                             # normal: use real distances
        s_all = cum_dist.copy()
    valid = s_all <= path_len                         # hide any label past the end of the curve
    projected_x = fx_path(s_all[valid])
    projected_y = fy_path(s_all[valid])

    print(f"\nDrawn curve length: {path_len:.1f} pixels")
    print(f"Labels placed on the curve: {int(valid.sum())} / {len(cum_dist)}")

    # 5) Show the final two-panel picture.
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    axes[0].imshow(img_rgb)                                                   # top: photo + curve
    axes[0].plot(x_path, y_path, color="cyan", linewidth=2, label=f"curve={path_len:.0f}px")
    axes[0].scatter(x_path[0], y_path[0], color="lime", s=90, label="TAIL START")
    axes[0].scatter(x_path[-1], y_path[-1], color="red", s=90, label="ROOT END")
    axes[0].set_title(f"{img_name}: your curve (tail -> root)")
    axes[0].legend(loc="upper right")
    axes[0].axis("off")

    axes[1].imshow(img_rgb)                                                   # bottom: photo + labels
    axes[1].plot(x_path, y_path, color="cyan", linewidth=1.8, alpha=0.9)
    axes[1].scatter(x_path[0], y_path[0], color="lime", s=90, label="TAIL START")
    axes[1].scatter(x_path[-1], y_path[-1], color="red", s=90, label="ROOT END")
    prev_x, prev_y = x_path[0], y_path[0]                                     # connect from the tail
    for idx, (x, y) in enumerate(zip(projected_x, projected_y), start=1):
        axes[1].plot([prev_x, x], [prev_y, y], linewidth=3, alpha=0.65)
        axes[1].scatter(x, y, color="yellow", s=55)
        axes[1].text(x + 6, y - 6, str(idx), color="yellow", fontsize=9)
        prev_x, prev_y = x, y
    axes[1].set_title(f"Real growth-ring labels projected | {int(valid.sum())}/{len(cum_dist)}")
    axes[1].legend(loc="upper right")
    axes[1].axis("off")

    plt.tight_layout()
    print("\nA window opened with the final picture. Close it to finish.")
    plt.show()                                        # show it (no saving)
    print("Done.")
