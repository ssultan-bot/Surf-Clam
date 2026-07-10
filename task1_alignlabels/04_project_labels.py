# 04_project_labels.py
# ----------------------------------------------------------------------
# STEP 4: Place the REAL growth-ring labels onto a straight line.
#
# Instead of tracing the curved hinge by hand, you just CLICK TWO POINTS on
# the photo:
#   - 1st click = the TAIL (where growth ends)
#   - 2nd click = the ROOT / HINGE (top of the shell)
# The program draws a straight line between them and "projects" each real
# Excel increment onto that line, starting from the tail.
#
# This step focuses on the MATH: it prints where every label lands.
# Step 5 makes the same thing into a nicer two-panel picture.
# Nothing is saved to disk.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# Window controls:  left-click = drop a point | R = reset | ENTER = confirm
# ----------------------------------------------------------------------

import os                                            # file-name helpers
import cv2                                           # read photo, window, mouse clicks
import numpy as np                                   # math on arrays
import pandas as pd                                  # read the Excel table
import matplotlib                                    # plotting library (core)
matplotlib.use("TkAgg")                              # show plots in a normal pop-up window (avoids a PyCharm backend bug)
import matplotlib.pyplot as plt                      # the drawing tools
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
    row = matched.iloc[0]                                              # first matching row
    i_cols = [c for c in df.columns if str(c).startswith("I")]         # columns I1, I2, ...
    i_series = row[i_cols].dropna()                                    # values, blanks removed
    increments = pd.to_numeric(i_series, errors="coerce").dropna().values.astype(float)  # to numbers
    if len(increments) == 0:                                          # no usable numbers -> stop
        raise ValueError("No valid increment labels found.")
    return increments                                                 # root -> tail


def pick_two_points(img_bgr):
    """Open a window, let the user click TWO points, return them in IMAGE coordinates."""
    h, w = img_bgr.shape[:2]                          # real image size (for scaling)
    pts_win = []                                       # the clicked points, in WINDOW pixels

    def render():                                      # build the picture to show
        canvas = cv2.resize(img_bgr.copy(), (config.WIN_W, config.WIN_H))   # fit to window
        if len(pts_win) == 2:                                              # if both points exist
            cv2.line(canvas, pts_win[0], pts_win[1], (0, 255, 255), 2)     # draw the straight line
        if len(pts_win) >= 1:                                              # 1st point = TAIL (green)
            cv2.circle(canvas, pts_win[0], 7, (0, 255, 0), -1)
            cv2.putText(canvas, "TAIL", (pts_win[0][0] + 8, pts_win[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(pts_win) >= 2:                                              # 2nd point = ROOT (red)
            cv2.circle(canvas, pts_win[1], 7, (0, 0, 255), -1)
            cv2.putText(canvas, "ROOT", (pts_win[1][0] + 8, pts_win[1][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(canvas, "Click TAIL then ROOT | R reset | ENTER confirm",
                    (20, config.WIN_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        return canvas

    def on_mouse(event, x, y, flags, param):           # runs on every mouse action
        if event == cv2.EVENT_LBUTTONDOWN:             # a left-click drops a point
            if len(pts_win) < 2:                        # only keep the first two clicks
                pts_win.append((x, y))
            cv2.imshow("Pick TAIL and ROOT", render())

    cv2.namedWindow("Pick TAIL and ROOT", cv2.WINDOW_NORMAL)                # make the window
    cv2.resizeWindow("Pick TAIL and ROOT", config.WIN_W, config.WIN_H)
    cv2.setMouseCallback("Pick TAIL and ROOT", on_mouse)                    # connect mouse handler
    cv2.imshow("Pick TAIL and ROOT", render())                             # show starting picture
    while True:                                        # keep window alive, watch keyboard
        key = cv2.waitKey(20)
        if key == 13 and len(pts_win) == 2:            # ENTER (with two points) -> done
            break
        if key in [ord("r"), ord("R")]:               # R -> reset
            pts_win.clear()
            cv2.imshow("Pick TAIL and ROOT", render())
    cv2.destroyAllWindows()                            # close the window

    # Convert WINDOW coordinates back to REAL IMAGE coordinates.
    pts = np.array([[px * w / config.WIN_W, py * h / config.WIN_H] for px, py in pts_win], dtype=float)
    return pts                                          # [[tail_x, tail_y], [root_x, root_y]]


def project_on_line(tail, root, cum_dist, fit_to_line):
    """Place each label distance onto the straight line from tail to root.

    Returns the x's and y's of the labels that fit, plus a 'valid' mask and
    the line length.
    """
    direction = root - tail                            # vector pointing tail -> root
    line_len = float(np.linalg.norm(direction))        # length of the line in pixels
    if line_len < 1:                                   # the two clicks were basically the same point
        raise ValueError("The two points are too close together.")
    unit = direction / line_len                        # unit vector (length 1) along the line
    if fit_to_line:                                    # optional: stretch labels to fill the line
        s_all = cum_dist / float(cum_dist[-1]) * line_len
    else:                                              # normal: use the real distances
        s_all = cum_dist.copy()
    valid = s_all <= line_len                          # hide any label past the end of the line
    projected = tail + np.outer(s_all[valid], unit)    # each point = tail + unit * distance
    return projected[:, 0], projected[:, 1], valid, line_len


# ----------------------------------------------------------------------
# MAIN program: runs only when you run THIS file directly in PyCharm.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    img_bgr = cv2.imread(config.IMG_PATH)             # read the photo
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_name = os.path.basename(config.IMG_PATH)

    # Read the REAL increments and turn them into distances from the tail.
    increments_root_to_tail = read_increments(config.LABEL_EXCEL, config.FILE_COL, img_name)
    increments_tail_to_root = increments_root_to_tail[::-1]            # reverse to tail -> root
    cum_dist = np.cumsum(increments_tail_to_root)                     # distance from tail per ring

    print("Click the TAIL point, then the ROOT point, then press ENTER...")
    pts = pick_two_points(img_bgr)                     # the user picks the two ends
    tail, root = pts[0], pts[1]

    # Project the real labels onto the straight line.
    px, py, valid, line_len = project_on_line(tail, root, cum_dist, config.FIT_LABELS_TO_DRAWN_CURVE)

    # Print the math result: where each label landed.
    print("=" * 80)
    print("PROJECTION RESULT (real labels on a straight line)")
    print("=" * 80)
    print(f"Image: {img_name}")
    print(f"Line length (tail -> root): {line_len:.1f} pixels")
    print(f"Total real label length:    {float(cum_dist[-1]):.1f} pixels")
    print(f"Labels placed on the line:  {int(valid.sum())} / {len(cum_dist)}")
    print("\nProjected label positions [x, y] on the image:")
    for i, (x, y) in enumerate(zip(px, py), start=1):
        print(f"  ring {i:2d}: x={x:7.1f}, y={y:7.1f}")

    # Show a simple picture: the line plus the numbered labels.
    plt.figure(figsize=(14, 5))
    plt.imshow(img_rgb)
    plt.plot([tail[0], root[0]], [tail[1], root[1]], color="cyan", linewidth=2, label="line")
    plt.scatter(tail[0], tail[1], color="lime", s=90, label="TAIL")
    plt.scatter(root[0], root[1], color="red", s=90, label="ROOT")
    plt.scatter(px, py, color="yellow", s=55, label="projected labels")
    for i, (x, y) in enumerate(zip(px, py), start=1):
        plt.text(x + 6, y - 6, str(i), color="yellow", fontsize=9)
    plt.title(f"Step 4: real labels projected onto the line | {int(valid.sum())}/{len(cum_dist)}")
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    print("\nA window opened with the projection. Close it to finish.")
    plt.show()                                        # show it (no saving)
    print("Done. Step 4 finished.")
