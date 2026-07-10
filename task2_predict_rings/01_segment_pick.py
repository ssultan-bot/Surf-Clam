# 01_segment_pick.py
# ----------------------------------------------------------------------
# STEP 1: Use the FastSAM AI to break the photo into shapes ("masks"),
#         then YOU click the one shape that is the clam shell.
#
# The chosen shape is saved so the next script can use it.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# (The first run downloads the FastSAM model, so it needs internet once.)
# Window: click a mask thumbnail, then press ENTER.
# ----------------------------------------------------------------------

import os                                            # file/folder toolbox
import config                                        # our settings

# These two lines must run BEFORE importing the AI library, so it knows where
# to keep its cache. setdefault = only set it if it isn't already set.
os.environ.setdefault("YOLO_CONFIG_DIR", config.YOLO_CACHE)

import cv2                                           # read/show images, mouse clicks
import numpy as np                                   # math on arrays
from ultralytics import FastSAM                      # the AI segmentation model

# Read the photo into memory.
img = cv2.imread(config.IMG_PATH)
if img is None:                                      # imread returns None if it failed
    raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # RGB copy (nicer colors for overlays)
h, w = img.shape[:2]                                 # image height and width

# ----------------------------------------------------------------------
# Run FastSAM to find all the shapes in the photo.
# ----------------------------------------------------------------------
print("Running FastSAM... (first time downloads the model)")
model = FastSAM(config.MODEL)                        # load the model
results = model(config.IMG_PATH, device=config.DEVICE, retina_masks=True,
                imgsz=config.IMGSZ, conf=config.CONF, iou=config.IOU,
                save=False, verbose=False)           # run it on our photo

# Pull the shapes (masks) out of the result and resize each to the photo size.
masks = results[0].masks.data.cpu().numpy()          # raw masks as arrays of 0/1
masks_resized = []                                   # will hold full-size masks
for m in masks:
    mr = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    masks_resized.append(mr)
n = len(masks_resized)                               # how many shapes we found
print(f"Total masks found: {n}")
if n == 0:
    raise ValueError("FastSAM found no masks. Try lowering CONF in config.py.")

# ----------------------------------------------------------------------
# Show all masks in a grid so you can click the clam shell.
# ----------------------------------------------------------------------
ROWS = (n + config.COLS - 1) // config.COLS          # how many rows the grid needs
selected = [None]                                    # remembers which mask you clicked

def build_canvas(highlight=None):
    """Draw the grid of mask thumbnails; outline the highlighted one in green."""
    canvas = np.zeros((ROWS * config.PH, config.COLS * config.PW, 3), dtype=np.uint8)
    for i, mask in enumerate(masks_resized):
        r, c = divmod(i, config.COLS)                # row, column for this thumbnail
        overlay = img_rgb.copy()                     # start from the photo
        # Tint the mask area orange so you can see what this shape covers.
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 180, 255]) * 0.5).astype(np.uint8)
        thumb = cv2.resize(overlay, (config.PW, config.PH))     # shrink to thumbnail size
        thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)          # back to BGR for OpenCV
        color = (0, 255, 100) if highlight == i else (0, 255, 255)   # green if selected
        cv2.putText(thumb, f"Mask {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if highlight == i:                                       # draw a green border if selected
            cv2.rectangle(thumb, (3, 3), (config.PW - 3, config.PH - 3), (0, 255, 100), 4)
        canvas[r * config.PH:(r + 1) * config.PH, c * config.PW:(c + 1) * config.PW] = thumb
    return canvas

def on_click(event, x, y, flags, param):
    """When you click, work out which thumbnail you hit and select it."""
    if event != cv2.EVENT_LBUTTONDOWN:               # only react to a left-click
        return
    idx = (y // config.PH) * config.COLS + (x // config.PW)   # which grid cell was clicked
    if 0 <= idx < n:
        selected[0] = idx
        print(f"Selected mask: {idx}")
        cv2.imshow("Step1: Click mask, press ENTER", build_canvas(idx))

cv2.namedWindow("Step1: Click mask, press ENTER", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Step1: Click mask, press ENTER", config.COLS * config.PW, ROWS * config.PH)
cv2.setMouseCallback("Step1: Click mask, press ENTER", on_click)
cv2.imshow("Step1: Click mask, press ENTER", build_canvas())
while True:                                          # wait until you pick one and press ENTER
    key = cv2.waitKey(20)
    if key == 13 and selected[0] is not None:        # 13 = ENTER
        break
cv2.destroyAllWindows()

# ----------------------------------------------------------------------
# Save the chosen mask for the next script.
# ----------------------------------------------------------------------
best_mask = masks_resized[selected[0]]               # the mask you picked
np.save(config.BEST_MASK_FILE, best_mask)            # save it to the intermediate folder
print(f"Saved chosen mask ({selected[0]}) to: {config.BEST_MASK_FILE}")
print("Done. Step 1 finished. Now run 02_draw_lines.py")
