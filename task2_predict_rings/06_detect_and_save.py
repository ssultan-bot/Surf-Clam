F# 06_detect_and_save.py
# ----------------------------------------------------------------------
# STEP 6: Find the dark-band valleys (= growth rings), draw the result,
#         and save the numbers to CSV files.
#
# A growth ring is a DARK dip in the smoothed brightness curve. find_peaks
# only finds high points, so we flip the curve (use -gray) to turn the dips
# into peaks and find those. Each one is a ring.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# (Run 01-05 first.)
# ----------------------------------------------------------------------

import os                                            # file/folder toolbox
import config                                        # our settings
os.environ.setdefault("MPLCONFIGDIR", config.MPL_CACHE)   # cache dir, set before matplotlib

import cv2                                           # read the image
import numpy as np                                   # math on arrays
import pandas as pd                                  # save CSV tables
import matplotlib                                    # plotting (core)
matplotlib.use("TkAgg")                              # open plots in a normal pop-up window
import matplotlib.pyplot as plt                      # the drawing tools
from scipy.signal import find_peaks                  # finds peaks (we use it to find valleys)

# ----------------------------------------------------------------------
# Load the photo, the chosen mask, and the profiles from step 5.
# ----------------------------------------------------------------------
img = cv2.imread(config.IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if not os.path.exists(config.PROFILES_FILE) or not os.path.exists(config.BEST_MASK_FILE):
    raise FileNotFoundError("Missing data. Run steps 01-05 first.")
best_mask = np.load(config.BEST_MASK_FILE)
profiles = list(np.load(config.PROFILES_FILE, allow_pickle=True))
if "gray_smooth" not in profiles[0]:                 # make sure step 5 ran
    raise KeyError("Smoothed data missing. Run 05_gaussian.py first.")

# ----------------------------------------------------------------------
# Find the valleys (rings) on each smoothed curve.
# ----------------------------------------------------------------------
for d in profiles:
    gray_smooth = d["gray_smooth"]
    gray_range = gray_smooth.max() - gray_smooth.min()          # how much brightness varies
    valleys, _ = find_peaks(-gray_smooth,                        # flip so dips become peaks
                            distance=config.MIN_DIST,            # min gap between rings
                            prominence=gray_range * config.PROMINENCE)  # how deep a dip must be
    d["valleys"] = valleys
    d["valley_x"] = d["x"][valleys]
    d["valley_y"] = d["y"][valleys]
    d["valley_gray"] = gray_smooth[valleys]
    print(f"Line {d['idx']}: {len(valleys)} growth rings detected")

# ----------------------------------------------------------------------
# Draw: masked photo with rings on top, one brightness chart per line.
# ----------------------------------------------------------------------
COLORS = ["cyan", "orange", "magenta", "lime", "yellow"]
n = len(profiles)
fig, axes = plt.subplots(n + 1, 1, figsize=(18, 5 * (n + 1)))
if n + 1 == 1:
    axes = [axes]

img_masked = img_rgb.copy()
img_masked[best_mask == 0] = 0                        # black out everything outside the shell
axes[0].imshow(img_masked)
for i, d in enumerate(profiles):
    c = COLORS[i % len(COLORS)]
    axes[0].plot(d["x"], d["y"], color=c, linewidth=2, label=f"Line {d['idx']} - {len(d['valleys'])} rings")
    axes[0].scatter(d["valley_x"], d["valley_y"], color="red", s=50, zorder=5)
    for vx in d["valley_x"]:
        axes[0].axvline(x=vx, color="red", linewidth=0.8, alpha=0.45)
axes[0].set_title("Detected growth rings (red markers = dark-band valleys)")
axes[0].legend(loc="upper right")
axes[0].axis("off")

for i, d in enumerate(profiles):
    c = COLORS[i % len(COLORS)]
    ax = axes[i + 1]
    ax.plot(d["x"], d["gray_smooth"], color=c, linewidth=2.0, label="smoothed brightness")
    ax.scatter(d["valley_x"], d["valley_gray"], color="red", s=80, zorder=5, label=f"{len(d['valleys'])} rings")
    for vx in d["valley_x"]:
        ax.axvline(x=vx, color="red", linewidth=0.8, alpha=0.4)
    ax.set_title(f"Line {d['idx']}: {len(d['valleys'])} rings "
                 f"(MIN_DIST={config.MIN_DIST}, PROMINENCE={config.PROMINENCE})")
    ax.set_xlabel("x (pixel column)")
    ax.set_ylabel("gray value (0-255)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(d["x"][0], d["x"][-1])

plt.tight_layout()
print("\nA window opened with the final result. Close it to save the CSVs.")
plt.show()

# ----------------------------------------------------------------------
# Save the numbers next to the original photo as CSV files.
# ----------------------------------------------------------------------
base = os.path.splitext(config.IMG_PATH)[0]          # photo path without ".jpg"
for d in profiles:
    pd.DataFrame({                                   # all pixels along the line
        "x": d["x"], "y": d["y"],
        "gray_raw": d["gray_raw"].astype(int),
        "gray_clahe": d["gray_clahe"].astype(int),
        "gray_smooth": d["gray_smooth"].round(2),
    }).to_csv(f"{base}_line{d['idx']}_pixels.csv", index=False)
    pd.DataFrame({                                   # just the ring positions
        "ring_no": np.arange(1, len(d["valley_x"]) + 1),
        "x": d["valley_x"], "y": d["valley_y"],
        "gray": d["valley_gray"].round(2),
    }).to_csv(f"{base}_line{d['idx']}_rings.csv", index=False)
    print(f"Saved line {d['idx']}: pixels CSV + rings CSV ({len(d['valley_x'])} rings)")

print("Done. Step 6 finished.")
