# 03_draw_curve.py
# ----------------------------------------------------------------------
# STEP 3: Draw the curve along the clam hinge BY HAND with the mouse.
#
# A window opens showing the photo. You hold the LEFT mouse button and drag
# along the hinge, starting at the TAIL and ending at the ROOT/HINGE. The
# program records every point your mouse passes through.
#
# Controls inside the window:
#   - Hold left mouse button and drag = draw the curve
#   - Press  R                        = reset / start over
#   - Press  ENTER                    = confirm and finish (need >= 5 points)
#
# After you finish, this script PRINTS the points the computer recorded and
# shows the curve on top of the photo. Nothing is saved to disk.
#
# HOW TO RUN IN PYCHARM:
#   Open this file and press the green "Run" button.
# ----------------------------------------------------------------------

import cv2                          # OpenCV: window, mouse events, drawing tools
import numpy as np                  # NumPy: store points as an array, do math
import matplotlib                   # the plotting library (core)
matplotlib.use("TkAgg")             # show plots in a normal pop-up window (avoids a PyCharm backend bug)
import matplotlib.pyplot as plt     # to show the final curve on the photo
import config                       # our settings file


def draw_curve_on_image(img_bgr):
    """Open a window, let the user draw a curve, and return the points.

    The returned points are in ORIGINAL IMAGE coordinates (not window
    coordinates), as a NumPy array shaped like [[x1, y1], [x2, y2], ...].
    """
    h, w = img_bgr.shape[:2]        # real image height and width (for scaling later)
    drawing = [False]               # on/off flag: is the mouse button held down?
    curve_pts_win = []              # collects points in WINDOW pixels as we draw

    def render_canvas():
        """Build the picture to show: the photo plus whatever we've drawn so far."""
        canvas = cv2.resize(img_bgr.copy(), (config.WIN_W, config.WIN_H))   # resize to window size
        if len(curve_pts_win) > 1:                                          # connect points with lines
            for i in range(1, len(curve_pts_win)):
                cv2.line(canvas, curve_pts_win[i - 1], curve_pts_win[i], (0, 255, 255), 2)
        if len(curve_pts_win) > 0:                                          # mark start (green) and end (red)
            cv2.circle(canvas, curve_pts_win[0], 7, (0, 255, 0), -1)        # green dot at start = TAIL
            cv2.putText(canvas, "TAIL START", (curve_pts_win[0][0] + 8, curve_pts_win[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(canvas, curve_pts_win[-1], 7, (0, 0, 255), -1)       # red dot at end = ROOT
            cv2.putText(canvas, "ROOT END", (curve_pts_win[-1][0] + 8, curve_pts_win[-1][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(canvas,                                                 # instructions at the bottom
                    "Draw from TAIL to ROOT | hold left mouse and drag | R reset | ENTER confirm",
                    (20, config.WIN_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        return canvas

    def on_mouse(event, x, y, flags, param):
        """Runs automatically every time the mouse does something in the window."""
        if event == cv2.EVENT_LBUTTONDOWN:          # button pressed -> start fresh curve
            drawing[0] = True
            curve_pts_win.clear()
            curve_pts_win.append((x, y))
            cv2.imshow("Draw curve from TAIL to ROOT", render_canvas())
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:   # dragging -> keep adding points
            curve_pts_win.append((x, y))
            cv2.imshow("Draw curve from TAIL to ROOT", render_canvas())
        elif event == cv2.EVENT_LBUTTONUP and drawing[0]:   # button released -> finish stroke
            drawing[0] = False
            curve_pts_win.append((x, y))
            print(f"Current curve points: {len(curve_pts_win)}")
            cv2.imshow("Draw curve from TAIL to ROOT", render_canvas())

    cv2.namedWindow("Draw curve from TAIL to ROOT", cv2.WINDOW_NORMAL)       # create the window
    cv2.resizeWindow("Draw curve from TAIL to ROOT", config.WIN_W, config.WIN_H)  # set its size
    cv2.setMouseCallback("Draw curve from TAIL to ROOT", on_mouse)          # connect our mouse handler
    cv2.imshow("Draw curve from TAIL to ROOT", render_canvas())             # show the starting picture

    while True:                                      # keep the window alive, listen for keys
        key = cv2.waitKey(20)                        # wait 20 ms for a key press
        if key == 13 and len(curve_pts_win) >= 5:    # 13 = ENTER, and we need >= 5 points
            break                                    # leave the loop -> done drawing
        if key in [ord("r"), ord("R")]:              # R = reset
            curve_pts_win.clear()
            print("Reset curve.")
            cv2.imshow("Draw curve from TAIL to ROOT", render_canvas())
    cv2.destroyAllWindows()                          # close the drawing window

    # Convert WINDOW coordinates back to REAL IMAGE coordinates.
    # The window was resized, so scale x by (w / WIN_W) and y by (h / WIN_H).
    curve_pts = np.array(
        [[px * w / config.WIN_W, py * h / config.WIN_H] for px, py in curve_pts_win],
        dtype=float
    )
    if len(curve_pts) < 5:                           # safety check
        raise ValueError("Too few valid curve points.")
    return curve_pts                                 # hand the points back


# ----------------------------------------------------------------------
# MAIN program: runs only when you run THIS file directly in PyCharm.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    img_bgr = cv2.imread(config.IMG_PATH)            # read the photo
    if img_bgr is None:                              # safety check
        raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # RGB copy for matplotlib

    print("A window opened. Draw from TAIL to ROOT, then press ENTER.")
    curve_pts = draw_curve_on_image(img_bgr)         # do the drawing

    # Show what the computer recorded.
    print("\n" + "=" * 80)
    print("CURVE POINTS THE COMPUTER RECORDED")
    print("=" * 80)
    print(f"Total number of points: {len(curve_pts)}")
    print("First 5 points [x, y] in image coordinates:")
    print(np.round(curve_pts[:5], 1))                # show just the first 5 to keep it short

    # Draw the recorded curve on top of the photo so you can check it.
    plt.figure(figsize=(14, 5))
    plt.imshow(img_rgb)                                                  # the photo
    plt.plot(curve_pts[:, 0], curve_pts[:, 1], color="cyan", linewidth=2)  # the curve
    plt.scatter(curve_pts[0, 0], curve_pts[0, 1], color="lime", s=90, label="TAIL START")  # start
    plt.scatter(curve_pts[-1, 0], curve_pts[-1, 1], color="red", s=90, label="ROOT END")   # end
    plt.title("Step 3: the curve you drew")
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    print("\nA window opened showing your curve. Close it to finish.")
    plt.show()                                       # show it (no saving)

    print("Done. Step 3 finished.")
