# ML growth-band line tracer

Traces the same measurement line as `02_draw_lines.py` (the classical
pipeline in the parent folder), but using a trained model instead of Otsu
thresholding + morphology. Built because the classical `segment()` reliably
fails whenever debris or a broken fragment is fused onto the shell in the
photo — no amount of threshold/morphology tuning fixes that, since the
information needed to separate them isn't really there in a single
intensity mask. A model that's actually seen examples of "this is shell,
that's debris" does much better.

## Why a line, not a mask

The first version of this predicted a whole-shell segmentation mask (like
the classical pipeline), auto-labeled via FastSAM. That approach hit a
ceiling: FastSAM's own masks were sometimes *also* fused to debris, so the
model just learned to reproduce the same failure. Switching the target to a
thin band drawn directly around a human-verified centerline was a strictly
stronger signal — it tells the model exactly where the line goes, not "the
general vicinity of a blob that usually contains the line."

## Pipeline

```
label_line.py           you click points tracing the true centerline on
                         images with no prediction yet -> data/manual_lines/

review_and_label.py      you review the CURRENT model's predictions on new
                         images -- accept (ENTER) or correct (click) ->
                         data/manual_lines/   (same output format/location)

lines_to_masks.py        converts manual_lines/*.npy into training pairs:
                         letterboxed image + thin band mask around the line
                         -> data/images/, data/masks/

train_unet.py            trains model.SmallUNet on data/images + data/masks
                         -> shell_unet.pt

infer.py                 loads shell_unet.pt, runs it on new images:
                         predict_prob()      raw per-pixel confidence
                         segment_ml()        thresholded binary mask
                         centerline_from_prob()  confidence-weighted centerline
                         bridge_gaps()        fills gaps where the model had
                                              no signal, flagged as placeholder
                         centerline_ml()      the whole thing end to end
```

`02_draw_lines.py`'s `segment_best_available()` automatically uses the
trained model if `shell_unet.pt` exists, falling back to the classical
`segment()` otherwise -- so the rest of that pipeline (root click, despike,
smoothing, etc.) is unchanged.

## The active-learning loop

This is the actual workflow for improving the model, and the reason none of
this needed painting hundreds of segmentation masks by hand:

1. `review_and_label.py --prioritize-worst` pre-scans a pool of unlabeled
   images from the full survey, ranks them by how little of the shell the
   current model is confident about, and shows you the worst ones first --
   these are the corrections that teach the model the most per click.
2. For each image, the model's current guess is drawn already. If it looks
   right, ENTER accepts it as-is (near-zero effort). If it's wrong, you
   click a corrected line instead.
3. `lines_to_masks.py` converts whatever's new into training pairs.
4. `train_unet.py` retrains on the (now larger) dataset.
5. Repeat. Each round costs a training run (~1-4 hrs on CPU depending on
   dataset size) but very little of your time, and specifically closes the
   gaps the model currently has, rather than randomly adding more of what
   it's already good at.

Progress across rounds so far (measured against a held-out set of
hand-drawn ground truth, mean nearest-point distance in pixels -- not just
eyeballed):

| Round | Training images | Mean error (px) |
|---|---|---|
| 1 (bad loss weighting) | 32 | 67.1 |
| 2 (fixed weighting) | 32 | 61.9 |
| 3 (+40 reviewed) | 77 | 49.4 |
| 4 (+30 more, prioritized) | 107 | *pending* |

## Key design decisions and why

- **`pos_weight` on the loss, but softened.** The band is only ~3% of
  pixels, so an unweighted loss lets the model minimize error by mostly
  predicting background (produces a short "stub" instead of a full line).
  The naive fix -- weight by the full inverse class frequency (~28x) --
  overcorrects into predicting broad, mis-centered blobs instead (false
  negatives get penalized so heavily that over-predicting becomes the safe
  bet). Using sqrt of the inverse frequency (~5x) balances recall against
  precision. This was found by comparing predictions directly against
  ground truth, not by eyeballing.
- **Confidence-weighted centerline extraction, not naive midpoint.** Taking
  `(mask.top + mask.bottom) / 2` per column is sensitive to a lopsided or
  over-thick predicted blob. Weighting by the model's raw confidence
  (before thresholding) is more robust -- pixels near the blob's fuzzy,
  low-confidence edges barely move the average.
- **Gap bridging is explicit, not silent.** Where the model has no signal
  at all (a real, still-occurring failure mode on some images), the line
  is bridged using the classical `segment()`'s rough extent as a "how far
  does the shell plausibly go" reference -- but those bridged points are
  flagged (`is_real=False`, drawn in red in the review tool) rather than
  presented identically to real model output. A measurement tool that
  can't tell you when it's guessing is worse than one that visibly admits
  it.
- **CPU-only, small model.** `SmallUNet` (base=16 channels) trains in a few
  hours on CPU with ~100 images. No GPU available in this environment;
  the model is deliberately small so that's still tractable.

## Known limitations

- A handful of images (e.g. ones with debris very tightly fused to the
  shell) remain poorly handled even after multiple review rounds -- these
  are known, not silent.
- ~107 training images is still small relative to the ~850 images of this
  type in the full survey. Remaining error looks consistent with needing
  more varied examples, not an architectural ceiling.
- Evaluation is against a held-out portion of the same actively-collected
  ground truth, not a fully independent test set -- true generalization to
  never-reviewed images is somewhat better estimated by the "real coverage"
  metric (how much of a random image's line the model is confident about)
  than by the pixel-error numbers alone.
