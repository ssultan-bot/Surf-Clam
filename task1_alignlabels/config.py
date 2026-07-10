# config.py
# ----------------------------------------------------------------------
# This file is the "settings" file for the whole project.
# Every other script (01, 02, 03, 04, 05) reads its settings from here.
# If you ever need to change a file path or a number, change it HERE ONLY,
# and all the other scripts will automatically use the new value.
#
# NOTE: This project does not save any result files to disk.
# Every script just shows its result on screen / prints it in the console.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 1) INPUT FILES  (the data this project reads)
# ----------------------------------------------------------------------

# Full path to the ONE clam hinge photo we want to work on.
# The "r" before the quotes means "raw string" -> it tells Python to treat
# the backslashes as normal characters (Windows paths use backslashes).
IMG_PATH = "/Users/sophia/Desktop/SurfClam/Surf Clam hinge images/NOAA Surf Clam Survey/Reader comparison selected images/Surf_Clam_1986_Summer_255_2.jpg"

# Full path to the Excel file that contains the growth-increment labels.
LABEL_EXCEL = "/Users/sophia/Desktop/SurfClam/Surf Clam hinge images/NOAA Surf Clam Survey/Reader comparison selected images/Results2.xlsx"

# The name of the column in the Excel file that stores the image file name.
# We use this column to find the row that belongs to our photo.
FILE_COL = "File"

# ----------------------------------------------------------------------
# 2) DISPLAY WINDOW SIZE  (only affects the drawing window in step 03)
# ----------------------------------------------------------------------

WIN_W = 1400  # width of the drawing window, in pixels
WIN_H = 420   # height of the drawing window, in pixels

# ----------------------------------------------------------------------
# 3) TUNING NUMBERS  (you can experiment with these)
# ----------------------------------------------------------------------

# How strongly to smooth the hand-drawn curve. Bigger = smoother (less shaky).
SMOOTH_PATH_SIGMA = 6

# Whether to stretch/shrink the label distances so they exactly fill the curve.
#   False = use the REAL pixel distances (scientifically correct)
#   True  = squeeze all labels onto the curve (only for a nice-looking slide)
FIT_LABELS_TO_DRAWN_CURVE = False
