# Surf Clam Growth Ring Analysis

Tools for measuring surf clam growth from hinge cross-section photos (NOAA
and Nantucket Surf Clam Survey data): aligning known age measurements onto a
photo, and automatically detecting growth rings from the image itself.

## Projects

- **[task1_alignlabels/](task1_alignlabels/)** — takes a clam's known yearly
  growth measurements (from an Excel file) and projects them onto a photo of
  its hinge, so you can see where each year's growth actually falls.
- **[task2_predict_rings/](task2_predict_rings/)** — the classical pipeline:
  segments the shell with FastSAM, you trace the growth line by hand, and it
  detects growth rings from the brightness profile along that line
  (Otsu thresholding + morphology under the hood).
- **[task2_predict_rings_ml/](task2_predict_rings_ml/)** — same goal, but
  replaces the classical segmentation with a small U-Net trained via an
  active-learning loop, for images where debris fused to the shell breaks
  the classical approach. See
  [task2_predict_rings_ml/ml/README.md](task2_predict_rings_ml/ml/README.md)
  for the model details and training pipeline.

## Setup

See [SETUP.md](SETUP.md) for installing Python and the required packages
(one-time). Each project folder also has its own README with exact
run-order instructions.

## Data

Source photos live under [Surf Clam hinge images/](Surf%20Clam%20hinge%20images/)
(NOAA Surf Clam Survey and the 2017 Nantucket survey). Large generated
artifacts (trained model weights, debug images, per-run intermediate data)
are gitignored rather than committed — regenerate them by running the
pipelines above.
