# 03_grayscale.py
# ----------------------------------------------------------------------
# STEP 3: Turn each drawn line into a clean pixel path, then read the
#         COLOR and the GRAYSCALE brightness along it.
#
# A computer can't look for "dark bands" in color easily, so we reduce the
# three color numbers (Red, Green, Blue) of each pixel to ONE brightness
# number (gray, 0-255). This script shows the comparison: the R, G, B
# values along the line versus the single gray value we will use.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# (Run 01 and 02 first.)
# ----------------------------------------------------------------------

import os                                            # file/folder toolbox
import config                                        # our settings
os.environ.setdefault("MPLCONFIGDIR", config.MPL_CACHE)   # cache dir, set before matplotlib

import cv2                                           # read the image
import numpy as np                                   # math on arrays
import matplotlib                                    # plotting (core)
matplotlib.use("TkAgg")                              # open plots in a normal pop-up window
import matplotlib.pyplot as plt                      # the drawing tools
from scipy.ndimage import gaussian_filter1d          # smooths a line of numbers
from scipy.interpolate import interp1d               # fills in points between samples
    
# ----------------------------------------------------------------------
# Load the photo (color + gray) and the lines you drew.
# ----------------------------------------------------------------------
img = cv2.imread(config.IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # color version (Red, Green, Blue)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # gray version (one brightness per pixel)
h, w = img.shape[:2]

if not os.path.exists(config.LINES_FILE):
    raise FileNotFoundError("No saved lines. Run 02_draw_lines.py first.")
lines = list(np.load(config.LINES_FILE, allow_pickle=True))

# Also load the mask chosen in step 1, so we can show the shell area.
if not os.path.exists(config.BEST_MASK_FILE):
    raise FileNotFoundError("No saved mask. Run 01_segment_pick.py first.")
best_mask = np.load(config.BEST_MASK_FILE)

# Make the masked pictures: keep the shell, black out everything else.
img_masked_color = img_rgb.copy()
img_masked_color[best_mask == 0] = 0                 # color version, background removed
img_masked_gray = img_gray.copy()
img_masked_gray[best_mask == 0] = 0                  # gray version, background removed


def process_line(win_pts):
    """Turn one hand-drawn line (window pixels) into a clean left-to-right path.

    Converts to image pixels, sorts by x, averages duplicate x's, fills in
    every x by interpolation, then smooths the SHAPE to remove hand shake.
    """
    orig = [(int(px * w / config.WIN_W), int(py * h / config.WIN_H)) for px, py in win_pts]
    xs = np.array([p[0] for p in orig])              # x values
    ys = np.array([p[1] for p in orig])              # y values
    order = np.argsort(xs)                           # left-to-right order
    xs, ys = xs[order], ys[order]
    xs_u, inv = np.unique(xs, return_inverse=True)   # unique x's
    ys_u = np.array([ys[inv == i].mean() for i in range(len(xs_u))])  # average y per x
    if len(xs_u) < 2:
        return xs_u, ys_u.astype(int)
    x_full = np.arange(xs_u[0], xs_u[-1] + 1)        # every x column from start to end
    y_full = interp1d(xs_u, ys_u, kind="linear")(x_full)            # fill in y's
    y_smooth = gaussian_filter1d(y_full, sigma=config.SMOOTH_SIGMA)  # smooth the shape
    return x_full, y_smooth.astype(int)


# ----------------------------------------------------------------------
# Build the data for every line: path + color channels + gray.
# ----------------------------------------------------------------------
profiles = []                                        # one dict per line
for i, line in enumerate(lines):
    x_arr, y_arr = process_line(line)                # clean pixel path
    y_arr = np.clip(y_arr, 0, h - 1)                 # keep inside the image
    x_arr = np.clip(x_arr, 0, w - 1)
    # Read the Red, Green, Blue and gray values at each point along the line.
    r = np.array([img_rgb[y_arr[j], x_arr[j], 0] for j in range(len(x_arr))], dtype=float)
    g = np.array([img_rgb[y_arr[j], x_arr[j], 1] for j in range(len(x_arr))], dtype=float)
    b = np.array([img_rgb[y_arr[j], x_arr[j], 2] for j in range(len(x_arr))], dtype=float)
    gray_raw = np.array([img_gray[y_arr[j], x_arr[j]] for j in range(len(x_arr))], dtype=float)
    profiles.append({"idx": i + 1, "x": x_arr, "y": y_arr,
                     "r": r, "g": g, "b": b, "gray_raw": gray_raw})
    print(f"Line {i + 1}: {len(x_arr)} points sampled, gray range [{gray_raw.min():.0f}, {gray_raw.max():.0f}]")

# ----------------------------------------------------------------------
# Figure 1: the mask area in color vs the same area in grayscale.
# One big figure with two side-by-side subplots.
# ----------------------------------------------------------------------
fig_img, ax_img = plt.subplots(1, 2, figsize=(16, 6))
ax_img[0].imshow(img_masked_color)                       # left: color shell
ax_img[0].set_title("Mask area (color)")
ax_img[0].axis("off")
ax_img[1].imshow(img_masked_gray, cmap="gray")           # right: grayscale shell
ax_img[1].set_title("Mask area (grayscale)")
ax_img[1].axis("off")
fig_img.tight_layout()

# ----------------------------------------------------------------------
# Figure 2: color channels vs the single gray value, along each line.
# ----------------------------------------------------------------------
n = len(profiles)
fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n))
if n == 1:                                           # make axes always a list
    axes = [axes]
for ax, d in zip(axes, profiles):
    ax.plot(d["x"], d["r"], color="red",   alpha=0.5, label="Red")
    ax.plot(d["x"], d["g"], color="green", alpha=0.5, label="Green")
    ax.plot(d["x"], d["b"], color="blue",  alpha=0.5, label="Blue")
    ax.plot(d["x"], d["gray_raw"], color="black", linewidth=2, label="Gray (used)")
    ax.set_title(f"Line {d['idx']}: color channels vs gray")
    ax.set_xlabel("x (pixel column)")
    ax.set_ylabel("value (0-255)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
print("\nTwo windows opened: (1) mask area color vs gray, (2) color channels vs gray. Close them to finish.")
plt.show()

# ----------------------------------------------------------------------
# Save the profiles for the next step (CLAHE).
# ----------------------------------------------------------------------
np.save(config.PROFILES_FILE, np.array(profiles, dtype=object), allow_pickle=True)
print(f"Saved {n} line profiles to: {config.PROFILES_FILE}")
print("Done. Step 3 finished. Now run 04_clahe.py")
