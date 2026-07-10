# 05_visualize.py
# ----------------------------------------------------------------------
# STEP 5: Make the final, nice-looking TWO-panel picture, using the REAL
#         labels and a straight line you define by clicking two points.
#
# You CLICK TWO POINTS on the photo:
#   - 1st click = the TAIL
#   - 2nd click = the ROOT / HINGE
# The program draws a straight line between them, projects the real Excel
# increments onto it, and shows two panels:
#   Top panel    = the photo + the line.
#   Bottom panel = the photo + the line + every label as a numbered dot.
#
# Step 4 does the same math but prints the numbers; this step makes the
# polished picture. Nothing is saved to disk.
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
    total_label_len = float(cum_dist[-1])

    print("Click the TAIL point, then the ROOT point, then press ENTER...")
    pts = pick_two_points(img_bgr)                     # the user picks the two ends
    tail, root = pts[0], pts[1]

    # Build the straight line and project the real labels onto it.
    direction = root - tail                            # vector tail -> root
    line_len = float(np.linalg.norm(direction))        # length of the line
    if line_len < 1:                                   # clicks too close together
        raise ValueError("The two points are too close together.")
    unit = direction / line_len                        # unit vector along the line
    if config.FIT_LABELS_TO_DRAWN_CURVE:               # optional: stretch labels to fill the line
        s_all = cum_dist / total_label_len * line_len
    else:                                              # normal: use the real distances
        s_all = cum_dist.copy()
    valid = s_all <= line_len                          # hide any label past the end of the line
    projected = tail + np.outer(s_all[valid], unit)    # each point = tail + unit * distance
    projected_x, projected_y = projected[:, 0], projected[:, 1]

    print(f"Line length: {line_len:.1f} px | labels placed: {int(valid.sum())}/{len(cum_dist)}")

    # Create the figure: 2 rows, 1 column of panels.
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    # ---- TOP PANEL: photo + line ----
    axes[0].imshow(img_rgb)                                                   # the photo
    axes[0].plot([tail[0], root[0]], [tail[1], root[1]], color="cyan", linewidth=2,
                 label=f"line length={line_len:.1f}px")
    axes[0].scatter(tail[0], tail[1], color="lime", s=90, label="TAIL")
    axes[0].scatter(root[0], root[1], color="red", s=90, label="ROOT")
    axes[0].set_title(f"{img_name}: straight line between your two points")
    axes[0].legend(loc="upper right")
    axes[0].axis("off")

    # ---- BOTTOM PANEL: photo + line + numbered labels ----
    axes[1].imshow(img_rgb)
    axes[1].plot([tail[0], root[0]], [tail[1], root[1]], color="cyan", linewidth=1.8, alpha=0.9)
    axes[1].scatter(tail[0], tail[1], color="lime", s=90, label="TAIL")
    axes[1].scatter(root[0], root[1], color="red", s=90, label="ROOT")
    for idx, (x, y) in enumerate(zip(projected_x, projected_y), start=1):     # draw each label
        axes[1].scatter(x, y, color="yellow", s=55)                          # yellow dot
        axes[1].text(x + 6, y - 6, str(idx), color="yellow", fontsize=9)     # its number
    axes[1].set_title(f"Real growth-ring labels on the line | {int(valid.sum())}/{len(cum_dist)}")
    axes[1].legend(loc="upper right")
    axes[1].axis("off")

    plt.tight_layout()                               # tidy spacing between panels
    print("A window opened with the final two-panel picture. Close it to finish.")
    plt.show()                                       # show it (no saving)
    print("Done. Step 5 finished.")
