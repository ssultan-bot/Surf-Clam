# Ventral-margin line tracing — methods, limitations, and future pathways

`02_draw_lines.py` traces a measurement line along a surf clam shell's ventral
(bottom) growth-ring band, from the root/hinge to the tail tip, for use in
age/growth analysis. This is a record of the current approach as of this
session, what it does and doesn't handle well, and where it could go next.

## Current method

Pipeline, in order (see `__main__` in `02_draw_lines.py`):

1. **`segment(gray)`** — isolate the shell from the background.
   CLAHE contrast enhancement → Otsu threshold → morphological open/close →
   connected components → pick the largest component whose fill ratio
   (area ÷ bounding-box area) is below 0.9, so a solid rectangular object
   (a reference card, a label) can't be mistaken for the shell → fill holes.

2. **`bottom_edge_path(mask)`** — for each column of the mask, take the
   lowest (max-y) foreground pixel. This traces the ventral margin directly
   from the segmentation mask rather than following the mask's outer
   contour, which structurally avoids "needle spikes" from shadowed
   rib-groove notches (a notch that doesn't reach the true bottom of its
   column is simply never seen).

3. **`despike(path)`** — removes breaks, debris, and glare artifacts from
   the traced edge, of any width. Fits a rolling, iteratively-reweighted
   *linear* trend (via Gaussian-kernel weighted least squares) to the path;
   flags columns that depart from it by more than an adaptive threshold;
   bridges flagged runs by straight interpolation between their nearest
   good neighbors. Using a local *line* rather than a local *mean* avoids
   misflagging genuine curvature (e.g. near a tapering tip) as a defect.

4. **`pick_root_click(img)`** — the user clicks the root (hinge/umbo) point
   in a resized preview window; ENTER confirms, R resets. This point both
   orients the path (root end = start) and anchors the next two steps.

5. **`trim_near(path, root_xy)`** — cuts the path at its closest approach to
   the clicked root point, discarding whichever side loops back around
   (handles a break/chip near a coiled hinge tip, where the mask's outer
   boundary keeps going around the outside of the coil past the damage).

6. **`relax_near_root(path, root_xy, gray)`** — pulls the stretch of path
   nearest the root inward, away from the edge of debris/chip material, and
   adds an explicit inward bulge (biased toward the brighter — i.e. shell —
   side) so that stretch approximates the implied ring curvature rather
   than hugging wherever the debris happens to end.

7. **`smooth(path)`** — median filter (removes single-point outliers) +
   Gaussian filter (sigma=18, tuned to flatten small per-rib jitter without
   cutting corners on real curvature) + cubic spline resampling to 600
   points.

8. **`orient_root_first` / `overlay`** — orient root-to-tail, draw the
   line with root (red) and tail (green) markers, save to
   `config.INTERMEDIATE_DIR`.

Ring detection itself has been removed from this file entirely — it now
only produces the traced line, per an earlier scope decision.

Validated by hand against 15 images total (5 + 10, run via scratch driver
scripts, not part of the committed pipeline) by visual inspection of each
result, not automated metrics.

## Limitations

- **Segmentation fails when the shell's mask merges with another bright
  object.** In 3 of 15 test images, the largest-connected-component mask
  fused the shell with an adjacent object — a second shell fragment, loose
  debris, or a blown-out (overexposed) patch of background — via a thin
  touching bridge. The existing fill-ratio filter only catches a *solid
  rectangle* merging in; it doesn't help once the merged shape is itself
  irregular.
  - Tried and rejected: erosion-based and watershed-based splitting. Both
    rely on finding the geometrically "thinnest neck" in the mask, but a
    clam shell's own tapering, curved shape creates neck-like points along
    its own length that are indistinguishable from a real object boundary
    by geometry alone — confirmed empirically (watershed fragmented
    already-correct masks just as often as it helped broken ones).
  - Tried and partially working: seeded flood-fill on the grayscale image
    from a second, shell-interior click, with a fixed tolerance relative to
    the seed pixel. This correctly excluded the junk in 2 of 3 cases, but a
    single fixed tolerance is in tension with itself: loose enough to
    tolerate a whole shell's natural growth-ring banding (which can gradate
    over long distances) and it starts leaking into background/junk again;
    tight enough to reliably stop at the junk boundary and it can stop
    short partway across a long, strongly-banded shell.
  - `cv2.grabCut` (foreground extraction via a proper color/intensity
    model, not a single-pixel-difference threshold) was identified as the
    most promising untested option, but timed out at full image resolution
    (some source images are up to 6720×3120) — untested at a downscaled
    resolution.

- **Every image requires a live, interactive click.** `pick_root_click`
  blocks on a human clicking a point and pressing ENTER; there's no
  headless/batch mode. This is a deliberate tradeoff (established earlier
  in this project) favoring practical accuracy over full automation, given
  the dataset is small (tens, not thousands, of images) — but it means the
  pipeline can't currently run unattended over the full survey archive.

- **Tuned pixel constants, not physical units.** `despike`'s sigma/threshold,
  `relax_near_root`'s bulge/probe distances, and `smooth`'s filter sizes are
  all raw pixel counts, not scaled to each image's actual resolution or the
  `2 mm` reference bar already visible in every photo. Images in this
  dataset range from ~2100 to ~6700 px wide; the current constants were
  tuned by eye against a handful of them and have not been stress-tested
  for images at the extremes of that range.

- **No quantitative validation yet.** All correctness checks so far have
  been visual (does the overlaid line look right against the photo), not
  compared against the existing manual reader measurements already present
  in this dataset (e.g. `Surf_Clam_1986_Summer_255_2_line1_pixels.csv`,
  `Lengths in mm.xlsx`, `Results2.xlsx`).

## Future pathways

Roughly in order of effort:

1. **Downscaled `grabCut`.** Resize to a few hundred px wide, run grabCut
   seeded from the existing Otsu mask (probable background/foreground) plus
   a small definite-foreground disc at the click, then upscale the
   resulting mask. Directly targets the tension found in flood-fill (models
   a distribution of shell-surface color/intensity rather than a fixed
   tolerance around one seed pixel) using a tool already installed, no new
   dependency. This is the natural next experiment.

2. **Scale-aware constants.** Anchor `despike`, `relax_near_root`, and
   `smooth`'s pixel constants to the `2 mm` scale bar already burned into
   each photo (detect its length via template/line matching, or read it
   from calibration spreadsheets already in the dataset) instead of raw
   pixel counts, so the same tuning holds across the full range of source
   image resolutions.

3. **Batch/headless mode.** Once segmentation is robust enough to not need
   a human-in-the-loop safety net, add a non-interactive path (root point
   supplied via a config/CSV instead of a live click) so the pipeline can
   run unattended over the full image archive, with the interactive click
   retained as a fallback for images that fail automatically.

4. **Quantitative validation against manual reader data.** Compare traced
   line length / ring positions against the existing reader-measured CSVs
   and spreadsheets already in the dataset, to move from "looks right" to
   a measured error rate — this would also surface any systematic bias
   (e.g. consistent over/under-tracing near the root) that visual spot
   checks might miss.

5. **Segment Anything (SAM) or similar pretrained learned segmentation**,
   if grabCut and scale-awareness still leave a meaningful failure rate. A
   pretrained click-to-mask model would likely outperform every classical
   approach tried so far, since it uses learned object understanding rather
   than pixel-color similarity or mask geometry — but it's a real
   architectural step up (new dependency, model weights, heavier inference)
   and is only worth it if the cheaper options don't close the gap.

6. **Train a small custom segmentation model on this dataset — likely the
   best long-run fix.** Every failure mode found so far (merged debris,
   overexposed patches, second shell fragments) comes down to the same root
   cause: every classical method here (Otsu, watershed, flood-fill,
   grabCut) only ever reasons about raw pixel intensity/geometry, with no
   notion of "what clam shell material actually looks like." A small model
   (e.g. a lightweight U-Net, or a fine-tuned segmentation backbone) trained
   directly on this survey's photos would learn that distinction instead of
   approximating it through thresholds and morphology, and should generalize
   across exposure, debris, and touching-object cases in one shot rather
   than needing a bespoke rule for each new failure mode as it turns up.
   The real cost is training data: with only ~15-30 source images, this
   needs either manually-corrected masks from the current pipeline's output
   (cheap — start from what `segment()` + a flood-fill/grabCut pass already
   produces, hand-fix the ~20% that are wrong) as labels, plus heavy data
   augmentation (rotation, crop, brightness/contrast jitter) to make a small
   dataset go further, and possibly transfer learning from an
   ImageNet-pretrained encoder rather than training from scratch. More
   upfront effort than any option above, but it directly targets the actual
   problem instead of working around it, and — unlike SAM — it would be
   specifically tuned to this exact imaging setup (lighting, background,
   shell material) rather than general-purpose.
