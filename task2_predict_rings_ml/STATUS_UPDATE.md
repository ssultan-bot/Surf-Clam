# Status Update: ML-Based Growth-Band Line Tracing

## Goal

Automatically trace the growth-ring measurement line on surf clam hinge
cross-section photos (from the NOAA Surf Clam Survey), replacing/augmenting
a classical image-processing pipeline that struggled on images where debris
or broken shell fragments are fused to the specimen in the photo.

## Approach

1. **No manual pixel-mask labeling.** Instead, lines were hand-drawn
   (a handful of clicks per image, tracing the true center of the growth
   band from hinge to tail) as ground truth.
2. **Small U-Net** trained from scratch to predict a thin band around that
   line, given the photo. Trained on CPU; each full training run takes
   roughly 1-4 hours depending on dataset size.
3. **Active-learning loop**: the model's own predictions are shown back for
   quick human review — accept if correct (one keystroke), correct with a
   few clicks if not. Each round of review both measures current quality
   and generates new training data targeted at the model's actual weak
   points, prioritized automatically by scanning a larger pool and
   surfacing the lowest-confidence predictions first.
4. **Quantitative evaluation**, not just visual inspection: predicted lines
   are compared against the hand-drawn ground truth via mean nearest-point
   distance (in pixels), plus a "real coverage" metric (what fraction of
   each predicted line is genuine model output vs. an explicitly-flagged
   placeholder for stretches where the model had no signal).

## Results so far

| Iteration | Training images | Mean error (px) | Median error (px) |
|---|---|---|---|
| Initial (naive loss weighting) | 32 | 67.1 | 55.7 |
| Corrected loss weighting | 32 | 61.9 | 46.5 |
| + 1st active-learning round | 77 | 49.4 | 42.2 |
| **+ 2nd active-learning round** | **107** | **44.5** | **40.1** |

Real coverage (genuine model signal, not placeholder) currently averages
**89%** across the evaluation set. Error has decreased with every
additional round of actively-collected training data, without any change
to model architecture — evidence the remaining gap is a data quantity/
diversity issue, not an architectural ceiling.

Engineering fixes found along the way by comparing model output directly
against ground truth (not eyeballing):
- An overly aggressive class-imbalance correction was causing the model to
  over-predict broad, mis-centered regions instead of a precise line —
  fixed by softening the loss weighting.
- Images where the model has no confident signal are explicitly bridged
  and flagged (not silently dropped or fabricated), so downstream use never
  mistakes a placeholder for a real measurement.

## Ring detection (first pass)

The classical pipeline's existing dark-band valley-detection step
(`03_grayscale.py` → `06_detect_and_save.py`) was reconnected to the new
line-tracing code (it had been disconnected) and no longer requires the
FastSAM dependency to run. A first validation of ring **count** against
known ages in the existing reader-measurement spreadsheet (10-image sample):
2/10 exact matches, mean absolute error 3.7 rings, with a systematic
undercounting bias. Diagnosis points to fixed (non-scale-aware) smoothing
parameters as the likely cause on large, densely-ringed specimens — the
immediate next technical target, not yet fixed.

## Current limitations

- Quality is not yet uniform: most images now track the true line closely,
  but a real minority (particularly a few images with debris very tightly
  fused to the shell) remain poorly handled. These are known and
  identifiable, not silent failures.
- Training set (107 images) is still small relative to the full survey
  (~850 images of this type exist).
- Ring-count detection is not yet validated to an acceptable accuracy;
  distance/mm validation against `Lengths in mm.xlsx` has not yet been
  attempted (requires per-image scale-bar detection for calibration).

## Next steps

- Make ring-detection smoothing scale-aware (physical units via each
  image's scale bar) to address the undercounting bias.
- Further active-learning rounds targeting the model's current
  lowest-confidence predictions.
- Distance/mm validation against `Lengths in mm.xlsx` and `Results2.xlsx`.
- Re-evaluate against ground truth after each round to track real progress
  quantitatively.
