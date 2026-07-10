# 05_gaussian.py
# ----------------------------------------------------------------------
# STEP 5: Smooth the brightness curve with a Gaussian filter.
#
# The CLAHE curve from step 4 is jagged: it has lots of tiny up-and-down
# wiggles from image noise. If we hunted for dark bands now, we'd find many
# fake ones. A Gaussian filter gently averages each point with its
# neighbors, flattening the noise while keeping the real dips (the rings).
#
# This script compares the CLAHE curve before vs after smoothing.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# (Run 01-04 first.)
# ----------------------------------------------------------------------

import os                                            # file/folder toolbox
import config                                        # our settings
os.environ.setdefault("MPLCONFIGDIR", config.MPL_CACHE)   # cache dir, set before matplotlib

import numpy as np                                   # math on arrays
import matplotlib                                    # plotting (core)
matplotlib.use("TkAgg")                              # open plots in a normal pop-up window
import matplotlib.pyplot as plt                      # the drawing tools
from scipy.ndimage import gaussian_filter1d          # the Gaussian smoothing filter

# ----------------------------------------------------------------------
# Load the profiles saved by step 4.
# ----------------------------------------------------------------------
if not os.path.exists(config.PROFILES_FILE):
    raise FileNotFoundError("No profiles. Run 03 and 04 first.")
profiles = list(np.load(config.PROFILES_FILE, allow_pickle=True))
if "gray_clahe" not in profiles[0]:                  # make sure step 4 ran
    raise KeyError("CLAHE data missing. Run 04_clahe.py first.")

# ----------------------------------------------------------------------
# Smooth each CLAHE curve with the Gaussian filter.
# ----------------------------------------------------------------------
for d in profiles:
    d["gray_smooth"] = gaussian_filter1d(d["gray_clahe"], sigma=config.GRAY_SIGMA)
    print(f"Line {d['idx']}: smoothed with sigma = {config.GRAY_SIGMA}")

# ----------------------------------------------------------------------
# Show the comparison: CLAHE curve vs smoothed curve.
# ----------------------------------------------------------------------
n = len(profiles)
fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n))
if n == 1:
    axes = [axes]
for ax, d in zip(axes, profiles):
    ax.plot(d["x"], d["gray_clahe"],  color="purple", linewidth=0.8, alpha=0.4, label="CLAHE (jagged)")
    ax.plot(d["x"], d["gray_smooth"], color="darkorange", linewidth=2.0, label=f"smoothed (sigma={config.GRAY_SIGMA})")
    ax.set_title(f"Line {d['idx']}: before vs after Gaussian smoothing")
    ax.set_xlabel("x (pixel column)")
    ax.set_ylabel("gray value (0-255)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
print("\nA window opened comparing jagged vs smoothed. Close it to finish.")
plt.show()

# ----------------------------------------------------------------------
# Save the updated profiles for the last step (valley detection).
# ----------------------------------------------------------------------
np.save(config.PROFILES_FILE, np.array(profiles, dtype=object), allow_pickle=True)
print(f"Saved smoothed profiles to: {config.PROFILES_FILE}")
print("Done. Step 5 finished. Now run 06_detect_and_save.py")
