# 02_read_labels.py
# ----------------------------------------------------------------------
# STEP 2: Open the Excel file and read the growth labels for our photo.
#
# Background:
#   A scientist measured the clam's yearly growth rings ("increments").
#   Each increment is a distance in pixels. They are stored in the Excel
#   file in columns named I1, I2, I3, ... in the order root -> tail.
#
# This script reads those numbers, reverses them (because later we draw
# the curve from tail -> root), prints them, and shows a quick bar chart.
#
# HOW TO RUN IN PYCHARM:
#   Open this file and press the green "Run" button.
# ----------------------------------------------------------------------

import os                 # built-in toolbox for file names
import numpy as np        # NumPy: fast math on lists of numbers
import pandas as pd       # pandas: reads Excel/CSV tables (like code-version of Excel)
import matplotlib                 # the plotting library (core)
matplotlib.use("TkAgg")           # show plots in a normal pop-up window (avoids a PyCharm backend bug)
import matplotlib.pyplot as plt   # for drawing the bar chart in the demo
import config             # our settings file


def read_increments(excel_path, file_col, image_name):
    """Read one image's growth increments from the Excel file.

    Returns the increments in their ORIGINAL Excel order (root -> tail).
    """
    df = pd.read_excel(excel_path)                   # load the whole sheet as a table
    if file_col not in df.columns:                   # make sure the name column exists
        raise ValueError(f"Cannot find column '{file_col}'. Available: {list(df.columns)}")

    # Find the row whose File cell exactly equals our image name.
    matched = df[df[file_col].astype(str) == image_name]
    # If no exact match, fall back to rows that just CONTAIN the name.
    if len(matched) == 0:
        matched = df[df[file_col].astype(str).str.contains(image_name, regex=False, na=False)]
    if len(matched) == 0:                            # still nothing? stop clearly
        raise ValueError(f"Cannot find label row for image: {image_name}")

    row = matched.iloc[0]                             # take the first matching row
    i_cols = [c for c in df.columns if str(c).startswith("I")]   # columns I1, I2, ...
    i_series = row[i_cols].dropna()                  # the values, with blanks removed
    # Turn them into plain numbers; anything non-numeric becomes NaN and is dropped.
    increments = pd.to_numeric(i_series, errors="coerce").dropna().values.astype(float)
    if len(increments) == 0:                         # no usable numbers? stop
        raise ValueError("No valid increment labels found.")
    return increments                                # root -> tail order


# ----------------------------------------------------------------------
# MAIN program: runs only when you run THIS file directly in PyCharm.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    img_name = os.path.basename(config.IMG_PATH)     # the photo's file name

    # Read the increments in Excel order (root -> tail).
    increments_root_to_tail = read_increments(config.LABEL_EXCEL, config.FILE_COL, img_name)

    # Reverse them so they go tail -> root (the direction we will draw later).
    increments_tail_to_root = increments_root_to_tail[::-1]

    # Running total of distances from the tail. [10,5,8] -> [10,15,23].
    cum_dist = np.cumsum(increments_tail_to_root)

    # The total length of all increments added together.
    total_label_len = float(np.sum(increments_tail_to_root))

    # Print a tidy report so you can see exactly what was read.
    print("=" * 80)
    print("LABEL INFO")
    print("=" * 80)
    print(f"Image: {img_name}")
    print(f"Number of increments (growth rings): {len(increments_root_to_tail)}")
    print("\nExcel order (root -> tail):")
    print(np.round(increments_root_to_tail, 3).tolist())
    print("\nReversed order we will use (tail -> root):")
    print(np.round(increments_tail_to_root, 3).tolist())
    print(f"\nTotal length of all increments: {total_label_len:.3f} pixels")
    print("Cumulative distance from the tail at each ring:")
    print(np.round(cum_dist, 3).tolist())

    # Draw a simple bar chart so you can SEE the increments, not just read them.
    plt.figure(figsize=(10, 4))                                  # make a chart canvas
    positions = np.arange(1, len(increments_tail_to_root) + 1)   # x positions: 1, 2, 3, ...
    plt.bar(positions, increments_tail_to_root)                  # one bar per increment
    plt.xlabel("Ring number (tail -> root)")                     # label the x axis
    plt.ylabel("Increment length (pixels)")                      # label the y axis
    plt.title(f"Growth increments for {img_name}")               # chart title
    plt.tight_layout()                                           # tidy spacing
    print("\nA bar chart window opened. Close it to finish.")
    plt.show()                                                   # show the chart (no saving)

    print("Done. Step 2 finished.")
