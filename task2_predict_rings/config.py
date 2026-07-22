# config.py
# ----------------------------------------------------------------------
# Settings for Task 2 (predict growth rings).
# Change paths or numbers HERE; every script reads them from this file.
# ----------------------------------------------------------------------

import os   # built-in toolbox for files and folders

# ----------------------------------------------------------------------
# 1) INPUT PHOTO
# ----------------------------------------------------------------------

# Full path to the clam hinge photo we work on.  Build it from this file so
# the project works after being moved to another computer or folder.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(
    PROJECT_DIR,
    "Surf Clam hinge images",
    "NOAA Surf Clam Survey",
    "Reader comparison selected images",
    "Surf_Clam_1986_Summer_255_2.jpg",
)

# ----------------------------------------------------------------------
# 2) WORKING FOLDER (for small in-between files passed between scripts)
# ----------------------------------------------------------------------

INTERMEDIATE_DIR = os.path.join(os.path.dirname(__file__), "intermediate")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)      # create it if it does not exist

BEST_MASK_FILE = os.path.join(INTERMEDIATE_DIR, "best_mask.npy")   # chosen mask (from step 01)
LINES_FILE     = os.path.join(INTERMEDIATE_DIR, "lines.npy")       # drawn lines (from step 02)
PROFILES_FILE  = os.path.join(INTERMEDIATE_DIR, "profiles.npy")    # per-line data, grown step by step

# ----------------------------------------------------------------------
# 3) FastSAM MODEL SETTINGS (the AI that finds shapes in the photo)
# ----------------------------------------------------------------------

MODEL  = "FastSAM-s.pt"   # the model file (auto-downloads on first run; needs internet once)
DEVICE = "cpu"            # "cpu" = no graphics card needed. Use "0" if you have a CUDA GPU.
IMGSZ  = 512              # size the model works at (bigger = slower, maybe more detail)
CONF   = 0.4              # how confident a shape must be to be kept
IOU    = 0.9              # overlap setting for removing duplicate shapes

# ----------------------------------------------------------------------
# 4) DISPLAY WINDOW SIZES
# ----------------------------------------------------------------------

WIN_W, WIN_H = 1400, 420   # size of the line-drawing window
COLS = 3                   # how many columns in the mask-picking grid
PW, PH = 600, 200          # size of each mask thumbnail in the grid

# ----------------------------------------------------------------------
# 5) TUNABLE PROCESSING NUMBERS (experiment with these)
# ----------------------------------------------------------------------

SMOOTH_SIGMA = 5      # smooth the hand-drawn line's SHAPE (bigger = less shaky)
CENTERLINE_SMOOTH_SIGMA = 90   # smooth the AUTOMATIC centerline before fitting (02_draw_lines.py)
                                # bigger = curve sweeps smoothly through the ribs instead of
                                # zigzagging into every growth ring's local dip
CLAHE_CLIP   = 2.0    # CLAHE strength: higher = stronger contrast boost
CLAHE_TILE   = 8      # CLAHE works in a grid of TILE x TILE squares
GRAY_SIGMA   = 3      # smooth the BRIGHTNESS curve before finding dark bands
MIN_DIST     = 15     # minimum gap (pixels) between two growth rings
PROMINENCE   = 0.08   # how obvious a dark band must be to count (smaller = more sensitive)

# ----------------------------------------------------------------------
# 6) CACHE FOLDERS (where the AI / plotting libraries store their cache)
#    These just avoid permission issues on some computers. Change freely.
# ----------------------------------------------------------------------

CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")
YOLO_CACHE = os.path.join(CACHE_DIR, "yolo")       # FastSAM/YOLO config cache
MPL_CACHE  = os.path.join(CACHE_DIR, "matplotlib") # matplotlib cache
os.makedirs(YOLO_CACHE, exist_ok=True)
os.makedirs(MPL_CACHE, exist_ok=True)
