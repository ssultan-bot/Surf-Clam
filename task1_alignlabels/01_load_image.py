# 01_load_image.py
# ----------------------------------------------------------------------
# STEP 1: Open the clam photo and look at what the computer actually "sees".
#
# A picture, to a computer, is just a big grid of numbers. Each tiny dot
# (pixel) has three numbers: how much Red, Green, and Blue it contains
# (each from 0 to 255). This script opens the photo, shows it, and PRINTS
# some of those numbers so you can see the image as the computer sees it.
#
# HOW TO RUN IN PYCHARM:
#   Open this file and press the green "Run" button (or right-click -> Run).
# ----------------------------------------------------------------------

import cv2          # OpenCV: the library we use to read and show images
import os           # built-in toolbox for file/folder names
import config       # our own settings file (config.py) from the same folder


def load_image(path):
    """Read an image file and return it in two color orders: BGR and RGB.

    OpenCV reads colors as Blue-Green-Red (BGR). Most other tools expect
    Red-Green-Blue (RGB), so we return both versions to be safe.
    """
    img_bgr = cv2.imread(path)                       # read the file into a grid of numbers
    if img_bgr is None:                              # imread returns None if it failed
        raise FileNotFoundError(f"Cannot read image: {path}")   # stop with a clear message
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)          # make an RGB copy
    return img_bgr, img_rgb                          # hand both back to whoever called us


# ----------------------------------------------------------------------
# The MAIN program. Everything inside this block runs only when you run
# THIS file directly (which is exactly what PyCharm's Run button does).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load the photo using the function above.
    img_bgr, img_rgb = load_image(config.IMG_PATH)

    # Work out the size of the image grid: height, width, number of color channels.
    h, w = img_rgb.shape[:2]                         # first two numbers are height and width
    img_name = os.path.basename(config.IMG_PATH)     # just the file name, no folder path

    # Print a tidy report about the image.
    print("=" * 80)
    print("IMAGE INFO")
    print("=" * 80)
    print(f"Image file: {img_name}")                 # which file we opened
    print(f"Width  (pixels): {w}")                   # how many pixels across
    print(f"Height (pixels): {h}")                   # how many pixels tall
    print(f"Shape (height, width, channels): {img_rgb.shape}")  # the raw grid size
    print(f"Data type of each number: {img_rgb.dtype}")         # usually 'uint8' = 0..255
    print(f"Smallest pixel value: {img_rgb.min()}")            # darkest value present
    print(f"Largest  pixel value: {img_rgb.max()}")            # brightest value present

    # Show the actual NUMBERS the computer sees for a tiny 3x3 corner of the
    # image (top-left). Each row is one pixel as [Red, Green, Blue].
    print("\nThe top-left 3x3 patch of pixels (each pixel is [R, G, B]):")
    print(img_rgb[0:3, 0:3])

    # Show the value of one single pixel in the middle of the image as an example.
    mid_y, mid_x = h // 2, w // 2                     # "//" is whole-number division
    print(f"\nThe single pixel at the center (row {mid_y}, col {mid_x}) is [R, G, B] = {img_rgb[mid_y, mid_x]}")

    # Finally, show the image in a pop-up window so you can compare numbers to the picture.
    print("\nA window opened with the photo. Click it and press any key to close.")
    cv2.imshow("Step 1: Clam hinge image", img_bgr)  # display it (BGR for OpenCV)
    cv2.waitKey(0)                                    # wait until you press a key
    cv2.destroyAllWindows()                           # close the window

    print("Done. Step 1 finished.")
