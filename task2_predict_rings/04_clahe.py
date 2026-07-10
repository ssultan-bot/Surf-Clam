# 04_clahe.py
# ----------------------------------------------------------------------
# STEP 4: Boost the contrast with CLAHE so faint growth rings stand out.
#
# CLAHE = "Contrast Limited Adaptive Histogram Equalization". In plain words:
# it brightens dark areas and darkens bright areas in small local patches,
# which makes the faint dark bands (the rings) much easier to see.
#
# NOTE: CLAHE is an IMAGE (2D) method, so we compute it on the whole gray
# image. But to match the rest of the project, we COMPARE its effect as a
# 1D curve: the gray values along your line, before vs after CLAHE.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# (Run 01, 02, 03 first.)
# ----------------------------------------------------------------------

import os                                            # file/folder toolbox
import config                                        # our settings
os.environ.setdefault("MPLCONFIGDIR", config.MPL_CACHE)   # cache dir, set before matplotlib

import cv2                                           # image + CLAHE
import numpy as np                                   # math on arrays
import matplotlib                                    # plotting (core)
matplotlib.use("TkAgg")                              # open plots in a normal pop-up window
import matplotlib.pyplot as plt                      # the drawing tools

# ----------------------------------------------------------------------
# Load the gray image and the profiles saved by step 3.
# ----------------------------------------------------------------------
img = cv2.imread(config.IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # the gray image CLAHE works on

if not os.path.exists(config.PROFILES_FILE):
    raise FileNotFoundError("No profiles. Run 03_grayscale.py first.")
profiles = list(np.load(config.PROFILES_FILE, allow_pickle=True))

# ----------------------------------------------------------------------
# Build the CLAHE tool and apply it to the whole gray image.
# ----------------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP,
                        tileGridSize=(config.CLAHE_TILE, config.CLAHE_TILE))
gray_clahe_img = clahe.apply(img_gray)               # contrast-enhanced gray image

# ----------------------------------------------------------------------
# For each line, read the enhanced gray value along the same path.
# ----------------------------------------------------------------------
for d in profiles:
    x_arr, y_arr = d["x"], d["y"]                    # the pixel path from step 3
    d["gray_clahe"] = np.array([gray_clahe_img[y_arr[j], x_arr[j]]
                                for j in range(len(x_arr))], dtype=float)
    print(f"Line {d['idx']}: CLAHE applied along {len(x_arr)} points")

# ----------------------------------------------------------------------
# Show the comparison: raw gray vs CLAHE gray, as a 1D curve per line.
# ----------------------------------------------------------------------
n = len(profiles)
fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n))
if n == 1:
    axes = [axes]
for ax, d in zip(axes, profiles):
    ax.plot(d["x"], d["gray_raw"],   color="gray",  linewidth=1.5, alpha=0.7, label="raw gray")
    ax.plot(d["x"], d["gray_clahe"], color="purple", linewidth=2.0, label="after CLAHE")
    ax.set_title(f"Line {d['idx']}: gray before vs after CLAHE "
                 f"(clip={config.CLAHE_CLIP}, tile={config.CLAHE_TILE})")
    ax.set_xlabel("x (pixel column)")
    ax.set_ylabel("gray value (0-255)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
print("\nA window opened comparing raw vs CLAHE. Close it to finish.")
plt.show()

# ----------------------------------------------------------------------
# Save the updated profiles for the next step (Gaussian smoothing).
# ----------------------------------------------------------------------
np.save(config.PROFILES_FILE, np.array(profiles, dtype=object), allow_pickle=True)
print(f"Saved CLAHE profiles to: {config.PROFILES_FILE}")
print("Done. Step 4 finished. Now run 05_gaussian.py")
