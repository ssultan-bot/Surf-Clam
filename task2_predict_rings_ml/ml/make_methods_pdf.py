"""
Generates a printable PDF of the corrected Methodology section, based
strictly on verified facts from the codebase (model.py, train_unet.py,
infer.py, lines_to_masks.py, label_line.py, review_and_label.py, the raw
train*.log files, and shell history for the actual review_and_label.py
invocations). Corrections applied after an initial draft mis-stated the
number of prediction-assisted annotation rounds (said three, verified two)
and conflated a train-split count with a total-dataset count.

Run:
    ./.venv/bin/python ml/make_methods_pdf.py
Output:
    ml/Methodology.pdf
"""
import os
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "Methodology.pdf")

MARGIN = 18
PAGE_W = 210 - 2 * MARGIN


class DocPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Methodology -- Automated Growth-Band Centerline Detection", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def h1(pdf, text):
    pdf.ln(3)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(PAGE_W, 7.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def h2(pdf, text):
    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 11.5)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(PAGE_W, 6.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def body(pdf, text, bold_lead=None):
    pdf.set_x(pdf.l_margin)
    if bold_lead:
        pdf.set_font("Helvetica", "B", 10.2)
        pdf.set_text_color(25, 25, 25)
        pdf.write(5.4, bold_lead + " ")
        pdf.set_font("Helvetica", "", 10.2)
        pdf.write(5.4, text)
        pdf.ln(5.4)
        pdf.ln(1)
        return
    pdf.set_font("Helvetica", "", 10.2)
    pdf.set_text_color(25, 25, 25)
    pdf.multi_cell(PAGE_W, 5.4, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def formula(pdf, text):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Courier", "", 9.5)
    pdf.set_text_color(30, 30, 30)
    pdf.set_fill_color(244, 244, 244)
    pdf.multi_cell(PAGE_W, 5.2, text, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)


def note(pdf, text):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 9.3)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(PAGE_W, 5.0, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def table(pdf, headers, rows, col_widths, font_size=8.5):
    pdf.ln(1)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", font_size)
    pdf.set_fill_color(228, 228, 228)
    for w_, txt in zip(col_widths, headers):
        pdf.cell(w_, 6.8, txt, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", font_size)
    for row in rows:
        pdf.set_x(pdf.l_margin)
        for w_, txt in zip(col_widths, row):
            pdf.cell(w_, 6.0, txt, border=1)
        pdf.ln()
    pdf.ln(2)


pdf = DocPDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(MARGIN, MARGIN, MARGIN)
pdf.add_page()

# ---- Title ----
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(15, 15, 15)
pdf.ln(20)
pdf.multi_cell(PAGE_W, 9, "Methodology", align="C")
pdf.set_font("Helvetica", "", 12.5)
pdf.set_text_color(70, 70, 70)
pdf.multi_cell(PAGE_W, 7,
               "Automated Growth-Band Centerline Detection in\nSurf Clam Hinge Cross-Sections",
               align="C")
pdf.ln(8)
pdf.set_font("Helvetica", "", 9.5)
pdf.set_text_color(110, 110, 110)
pdf.multi_cell(PAGE_W, 5.5,
               "Every quantitative claim in this document was verified directly against source "
               "files (model.py, train_unet.py, infer.py, lines_to_masks.py, label_line.py, "
               "review_and_label.py), raw training logs (train.log through train7.log), and "
               "recorded shell history for tool invocations -- not reconstructed from memory. "
               "An earlier draft of this section contained two errors (an incorrect count of "
               "annotation rounds, and a conflated train-split/total-dataset figure); both are "
               "corrected here, with the corrections noted where relevant.",
               align="C")

pdf.add_page()

# ============================================================
h1(pdf, "III. Methodology")

h2(pdf, "A. Problem Formulation")
body(pdf,
     "The task addressed is the automated localization of the growth-band centerline in "
     "cross-sectional photographs of surf clam (Spisula solidissima) hinge sections, a "
     "measurement traditionally performed manually to support age and growth-rate estimation. "
     "Given an RGB image I of a hinge cross-section, the objective is to produce a "
     "one-dimensional curve, ordered from the hinge (umbo) to the ventral margin, that traces "
     "the visual center of the growth-band ridge structure. This is formulated as a per-pixel "
     "binary segmentation problem -- predicting a narrow band around the true centerline -- "
     "followed by a deterministic post-processing stage that collapses the predicted band into "
     "a single-pixel-wide curve.")
body(pdf,
     "This formulation was arrived at after two alternative formulations were implemented and "
     "found inadequate: (1) whole-shell segmentation via classical thresholding, with the "
     "centerline derived geometrically from the segmented region, and (2) whole-shell "
     "segmentation masks generated automatically via a pretrained promptable segmentation model "
     "(FastSAM) without human-verified ground truth. Both were found to fail specifically on "
     "specimens where debris or fragment material was fused to the shell in the photograph, "
     "motivating the direct-centerline-supervision approach described below.")

h2(pdf, "B. Dataset and Annotation Protocol")
body(pdf,
     "Source imagery was drawn from the NOAA Surf Clam Survey archive, comprising RGB "
     "photographs of hinge cross-sections at native resolutions ranging from approximately "
     "2100 to 6700 pixels in width, captured across multiple survey years (1986, 2008, "
     "2011-2016) plus a curated 'reader comparison' subset. The full accessible pool across "
     "these sources totals 815 candidate images of this photographic type.")
body(pdf,
     "Ground-truth annotation was performed using two purpose-built interactive tools rather "
     "than a general-purpose annotation platform:")
pdf.set_font("Helvetica", "", 10.2)
pdf.set_x(pdf.l_margin + 4)
pdf.multi_cell(PAGE_W - 4, 5.4,
               "1. Direct annotation: for images with no existing model prediction, the "
               "annotator placed a sequence of point clicks tracing the centerline from the "
               "hinge to the ventral tail, typically 6-15 points per image.")
pdf.set_x(pdf.l_margin + 4)
pdf.multi_cell(PAGE_W - 4, 5.4,
               "2. Prediction-assisted review: for subsequent rounds, the current model's "
               "prediction was rendered on the image prior to annotation. The annotator either "
               "accepted the prediction unmodified or supplied a corrective sequence of clicks, "
               "which replaced the model output as ground truth. Predicted segments for which "
               "the model produced no confident output were visually distinguished from "
               "confident segments, directing annotator attention toward the regions of highest "
               "correction value.")
pdf.ln(1)
body(pdf,
     "At the time of the evaluation reported in Section H, the annotated set comprised 107 "
     "images: 37 from an initial direct-annotation phase (drawn from the curated "
     "reader-comparison subset), followed by two prediction-assisted rounds contributing 40 "
     "and 30 images respectively (77 and 107 cumulative total). The first prediction-assisted "
     "round used uniform random sampling from the unannotated pool; the second used a "
     "coverage-prioritized strategy, in which a larger candidate pool (150 images) was "
     "pre-scored by the fraction of the classically-estimated shell extent for which the "
     "current model produced confident output, with the lowest-coverage candidates selected "
     "for annotation first. Both the random and coverage-prioritized sampling modes are "
     "implemented as selectable options in the same tool; use of the coverage-prioritized mode "
     "for the second round is confirmed by the recorded command invocation.")
note(pdf,
     "Correction from an earlier draft: this section previously stated 'three subsequent "
     "prediction-assisted rounds.' Cross-checking training-log headers (train6.log: 77 total "
     "images; train7.log: 107 total images) against recorded tool invocations confirms exactly "
     "two such rounds (contributing 40 and 30 images), not three.")

h2(pdf, "C. Ground-Truth Target Construction")
body(pdf,
     "Point-based annotations were converted into a rasterized supervision target via cubic "
     "B-spline interpolation (scipy.interpolate.splprep/splev; degree k = min(3, n-1) for n "
     "annotated points; smoothing factor s=0) through the annotated points, resampled to 300 "
     "curve points. A binary band mask was then rasterized by drawing this curve as a polyline "
     "of thickness 2w, where the half-width w = 0.02 x max(H, W) is set proportionally to the "
     "longer image dimension. Both the source image and the resulting mask were resized with "
     "aspect-ratio preservation and zero-padded to a fixed 512x512 resolution prior to network "
     "training.")

pdf.add_page()

h2(pdf, "D. Network Architecture")
body(pdf,
     "The centerline predictor is a convolutional encoder-decoder network following the U-Net "
     "design [1], comprising four downsampling stages, a bottleneck, and four upsampling "
     "stages connected via skip connections between corresponding-resolution encoder and "
     "decoder features. Each stage consists of two 3x3 convolutions (same-padding) with batch "
     "normalization and ReLU activation. Channel width follows the standard doubling "
     "convention (16, 32, 64, 128, 256 at the bottleneck), and spatial downsampling/upsampling "
     "is performed via 2x2 max-pooling and 2x2-stride transposed convolution, respectively. "
     "The network takes a 512x512x3 input and produces a 512x512x1 output map of unnormalized "
     "logits, with no output activation applied within the network itself. The complete "
     "network comprises 1,944,049 trainable parameters, verified by direct enumeration; no "
     "pretrained weights are used at any stage -- the model is trained from random "
     "initialization.")
note(pdf, "Full layer-by-layer architecture, verified tensor shapes at every stage, and a "
          "parameter-count breakdown by component are provided in the companion document, "
          "'U-Net Architecture'.")

h2(pdf, "E. Training Procedure")
body(pdf,
     "The network was trained to minimize a compound objective combining pixel-wise binary "
     "cross-entropy and a soft Dice coefficient loss:")
formula(pdf, "L  =  L_BCE( sigmoid(logits), target ; lambda )  +  L_Dice( sigmoid(logits), target )")
body(pdf,
     "Because target pixels constitute a minority class (empirically 3.3-3.5% of pixels "
     "across training images, depending on the round), the positive class was upweighted in "
     "the BCE term by a factor")
formula(pdf, "lambda  =  sqrt( (1 - p) / p )")
body(pdf,
     "where p is the measured foreground pixel fraction on the training split for the given "
     "run (yielding lambda in the range 5.3-5.4 across the rounds using this weighting). This "
     "weighting was arrived at empirically: an initial experiment using the full "
     "(non-square-rooted) inverse-frequency weight (lambda = 27.9) was found, by direct "
     "comparison of model predictions against held-out ground truth, to induce systematic "
     "over-prediction -- thick, spatially offset mask regions rather than a precise centerline "
     "-- consistent with the loss function's asymmetric penalty structure making false "
     "negatives on the rare class disproportionately costly relative to false positives on the "
     "majority class.")
body(pdf,
     "Optimization used Adam (learning rate 1e-3) with cosine-annealed learning rate decay "
     "over 150 epochs, batch size 4. A random 15% of the annotated set was held out for "
     "validation at each training run (reshuffled, fixed random seed, when the annotated set "
     "size changed between rounds); the checkpoint corresponding to the lowest validation loss "
     "across the run was retained. Training-time data augmentation comprised random "
     "horizontal and vertical flips (p=0.5 each), random rotation by a multiple of 90 degrees "
     "(uniform over four orientations), and multiplicative/additive brightness jitter "
     "(gain ~ Uniform(0.8, 1.2), bias ~ Uniform(-20, 20), applied with p=0.5), applied "
     "identically to image and mask for geometric transforms and to the image only for "
     "photometric jitter.")

table(pdf,
      ["Run", "Total images (train+val)", "Target", "Loss weighting"],
      [
          ["train.log", "38 (33+5)", "whole-shell (abandoned)", "unweighted"],
          ["train2.log", "34 (29+5)", "whole-shell (abandoned)", "unweighted"],
          ["train3.log", "37 (32+5)", "centerline band", "unweighted"],
          ["train4.log", "37 (32+5)", "centerline band", "lambda=27.9"],
          ["train5.log", "37 (32+5)", "centerline band", "lambda=5.3"],
          ["train6.log", "77 (66+11)", "centerline band", "lambda=5.4"],
          ["train7.log", "107 (91+16)", "centerline band", "lambda=5.3 (current)"],
      ],
      [30, 42, 55, 43], font_size=8)

pdf.add_page()

h2(pdf, "F. Iterative (Human-in-the-Loop) Training Set Expansion")
body(pdf,
     "Because initial models trained on the 37-image direct-annotation set exhibited two "
     "identifiable failure modes -- incomplete prediction coverage over portions of the shell, "
     "and (after correcting for the former via loss reweighting) systematic centerline "
     "mislocalization on a subset of images -- model improvement was pursued through an "
     "iterative annotation-training cycle rather than through architecture modification alone.")
body(pdf,
     "Five training runs were conducted in total for the direct-centerline-supervision "
     "approach. The first three used the fixed 37-image annotated set and differed only in "
     "loss weighting (Section E): unweighted, then lambda=27.9, then lambda=5.3, motivated by "
     "the failure modes described above. The remaining two runs followed each "
     "prediction-assisted annotation round (Section B), using the resulting 77-image and "
     "107-image sets respectively. This process is characterized as a form of human-in-the-loop "
     "active learning. Sample-efficiency gains from coverage-prioritized versus random "
     "candidate selection were not isolated or measured against each other.")
note(pdf,
     "Correction from an earlier draft: this section previously described 'four cycles... "
     "expanding the annotated set from 32 to 37... to 77 to 107,' which incorrectly presented "
     "32 (a train-split count) as a sequential total-dataset size. 32 was never a total "
     "annotated-set size; it is the training portion of the 37-image set (32 train + 5 "
     "validation = 37 total).")

h2(pdf, "G. Inference and Post-Processing")
body(pdf,
     "At inference, the trained network is evaluated on a letterboxed 512x512 representation "
     "of the input image, and the output logit map is resized (bilinear interpolation) back to "
     "native resolution after sigmoid activation, yielding a dense per-pixel confidence map "
     "p(x,y) in [0,1]. A binary mask is obtained by thresholding at p > 0.5, retaining only the "
     "largest 8-connected component and filling enclosed holes. Rather than deriving the "
     "centerline as the unweighted vertical midpoint of this binary mask per image column -- a "
     "formulation found to be sensitive to asymmetric or over-extended mask regions -- the "
     "centerline y-coordinate at each column x is computed as the confidence-weighted mean:")
formula(pdf, "y_hat(x)  =  sum_y[ y * p(x,y) * mask(x,y) ]  /  sum_y[ p(x,y) * mask(x,y) ]")
body(pdf,
     "restricted to the column's binary mask extent, with a fallback to the unweighted "
     "midpoint where the denominator is degenerate. The resulting curve is passed through an "
     "outlier-rejection filter (iteratively reweighted local linear regression, rejecting "
     "points deviating from a locally-fit trend beyond a fixed threshold) to remove residual "
     "local artifacts.")
body(pdf,
     "Columns for which the network produces no confident foreground prediction -- an observed "
     "residual failure mode, particularly on images with debris tightly conforming to the "
     "shell surface -- are not silently omitted. Instead, the plausible horizontal extent of "
     "the shell is separately estimated using a classical Otsu-threshold-based segmentation "
     "(used only to bound the column range, not to inform vertical position), and any "
     "resulting gap in the network's prediction is bridged: linearly interpolated where "
     "bounded on both sides by confident predictions, or extrapolated as a constant value "
     "where the gap extends to the boundary of the plausible extent. Each output point retains "
     "a boolean flag indicating whether it derives from direct network confidence or from this "
     "bridging procedure.")

pdf.add_page()

h2(pdf, "H. Evaluation Protocol")
body(pdf,
     "Quantitative evaluation used a symmetric nearest-point (Chamfer) distance between each "
     "predicted centerline and its corresponding human-annotated reference curve (the latter "
     "resampled via the same spline procedure as Section C), computed independently in pixel "
     "units for each of the 107 annotated images using the model checkpoint retained from the "
     "final training run (train7.log).")

table(pdf,
      ["Metric", "Value"],
      [
          ["Images evaluated", "107 (0 empty-mask failures)"],
          ["Mean directed error (pred -> ground truth)", "39.74 px"],
          ["Median directed error", "31.12 px"],
          ["Mean directed error, as % of image diagonal", "1.07%"],
          ["Mean symmetric Chamfer distance", "30.56 px"],
          ["Median symmetric Chamfer distance", "23.60 px"],
          ["Mean non-placeholder (real) coverage", "87.4%"],
          ["Worst-case directed error", "223.1 px (debris-fused specimen)"],
          ["Held-out validation loss (final run)", "0.360"],
      ],
      [110, 60], font_size=8.5)

body(pdf,
     "The single worst-performing image in the evaluation set exhibited a directed error of "
     "223 pixels, corresponding to a case of debris material fused directly onto the shell "
     "surface -- the same failure category motivating the annotation-driven approach -- for "
     "which the network produced confident output over only a minority of the image.")
body(pdf,
     bold_lead="Limitation of this protocol, stated explicitly:",
     text=
     "the reported evaluation set is the full annotated set used across training rounds and is "
     "not disjoint from the images on which the retained model was trained (a random 15% "
     "subset was held out during each individual training run, but the cumulative evaluation "
     "above spans both the final run's training and validation partitions). These figures "
     "should therefore be interpreted as a measure of fit to the accumulated human-verified "
     "reference set rather than as an estimate of generalization performance to entirely "
     "unannotated imagery. The held-out validation loss from the final training run (0.360, "
     "reported above) is provided as an auxiliary -- though not directly comparable in scale "
     "to the pixel-distance metric -- indicator of generalization. No evaluation against an "
     "independent, never-annotated test partition has been performed at the time of writing.")

pdf.ln(4)
h2(pdf, "Reference")
body(pdf,
     "[1] O. Ronneberger, P. Fischer, and T. Brox, \"U-Net: Convolutional Networks "
     "for Biomedical Image Segmentation,\" in Proc. MICCAI, 2015.")

pdf.output(OUT_PATH)
print(f"Saved: {OUT_PATH}")
