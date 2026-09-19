"""
Generates a printable PDF describing the U-Net architecture used for
growth-band line detection, based directly on model.py (verified by reading
the file and by actually running it to get exact shapes/parameter counts --
nothing here is estimated).

Run:
    ./.venv/bin/python ml/make_unet_pdf.py
Output:
    ml/UNet_Architecture.pdf
"""
import os
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "UNet_Architecture.pdf")

MARGIN = 18
PAGE_W = 210 - 2 * MARGIN  # A4 width minus margins, mm


class DocPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "U-Net Architecture -- Growth-Band Line Detection Model", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def h1(pdf, text):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(PAGE_W, 8, text)
    pdf.ln(1)


def h2(pdf, text):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(PAGE_W, 7, text)
    pdf.ln(1)


def body(pdf, text):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(PAGE_W, 5.6, text)
    pdf.ln(1)


def bullet(pdf, text):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(30, 30, 30)
    pdf.set_x(pdf.l_margin + 4)
    pdf.multi_cell(PAGE_W - 4, 5.6, f"-  {text}")


def code(pdf, text):
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.set_fill_color(244, 244, 244)
    pdf.multi_cell(PAGE_W, 4.6, text, fill=True)
    pdf.ln(1)


def note(pdf, text):
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(PAGE_W, 5.2, text)
    pdf.ln(1)


pdf = DocPDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(MARGIN, MARGIN, MARGIN)
pdf.add_page()

# ---- Title page content ----
pdf.set_font("Helvetica", "B", 20)
pdf.set_text_color(15, 15, 15)
pdf.ln(30)
pdf.multi_cell(PAGE_W, 10, "U-Net Architecture", align="C")
pdf.set_font("Helvetica", "", 13)
pdf.set_text_color(70, 70, 70)
pdf.multi_cell(PAGE_W, 8, "Growth-Band Centerline Detection Model", align="C")
pdf.ln(6)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(PAGE_W, 6,
               "Source: task2_predict_rings_ml/ml/model.py\n"
               "All figures below (shapes, parameter counts) were verified by directly\n"
               "reading and running this file, not estimated from memory.",
               align="C")
pdf.add_page()

# ---- Overview ----
h1(pdf, "1. Overview")
body(pdf,
     "This model takes a color photograph of a clam shell cross-section (letterboxed to a "
     "512x512 square) and outputs a same-resolution map of per-pixel scores: how likely each "
     "pixel is to lie on the true growth-band centerline. It is a U-Net -- an encoder-decoder "
     "convolutional network with skip connections, a design originally introduced by Ronneberger, "
     "Fischer & Brox (2015) for biomedical image segmentation, and now used extremely broadly "
     "for any task requiring a per-pixel prediction aligned to the input image.")
body(pdf,
     "The encoder repeatedly shrinks the image spatially while increasing the number of learned "
     "feature channels, trading spatial resolution for a richer feature representation at each "
     "scale. The decoder mirrors this in reverse, growing the spatial size back to the original "
     "resolution. At each decoder step, it is given a copy of the encoder's features from the "
     "matching resolution (a skip connection), so fine spatial detail lost during downsampling "
     "does not have to be reconstructed from a small, low-resolution bottleneck alone.")
note(pdf,
     "Provenance note: this implementation was written from general knowledge of the standard "
     "U-Net pattern, not copied from a specific paper's source code or a specific tutorial. It "
     "is a commonly used simplified variant -- same-padding convolutions and batch normalization "
     "-- rather than a reproduction of the original 2015 paper's exact specification (which used "
     "unpadded convolutions with cropping, and predates the widespread use of batch norm).")

# ---- Input/Output ----
h1(pdf, "2. Input / Output Contract")
bullet(pdf, "Input: tensor of shape (N, 3, 512, 512) -- N images, 3 RGB channels, 512x512 pixels.")
bullet(pdf, "Output: tensor of shape (N, 1, 512, 512) -- one raw logit per pixel.")
body(pdf, "")
body(pdf,
     "The output is a raw, unbounded logit, not a probability -- no sigmoid is applied inside "
     "the model. A sigmoid is applied afterward: implicitly inside the training loss function "
     "(nn.BCEWithLogitsLoss applies it internally for numerical stability), or explicitly during "
     "inference (torch.sigmoid() is called on the output in the separate inference code).")

# ---- The building block ----
h1(pdf, "3. The Repeated Building Block: conv_block")
body(pdf,
     "Every stage of the network, encoder and decoder alike, is built from the same function, "
     "which returns two convolution layers in sequence:")
code(pdf,
     'def conv_block(cin, cout):\n'
     '    return nn.Sequential(\n'
     '        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),\n'
     '        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),\n'
     '    )')
bullet(pdf, "Conv2d(cin, cout, 3, padding=1): a 3x3 convolution, cin input channels to cout "
             "output channels. padding=1 keeps the output the same height/width as the input "
             "(a 3x3 kernel with no padding would shrink each side by 1 pixel).")
bullet(pdf, "BatchNorm2d(cout): normalizes each channel's activations across the batch, then "
             "applies a learned per-channel scale and shift. Stabilizes and speeds up training.")
bullet(pdf, "ReLU(inplace=True): the activation max(0, x). Without a non-linearity, stacking "
             "convolutions would mathematically collapse into one linear operation regardless of "
             "depth. inplace=True overwrites the input tensor's memory rather than allocating a "
             "new one -- a memory optimization with no effect on the output values.")
body(pdf, "Two convolutions per block (rather than one) is standard practice in U-Net-style "
          "networks, giving more representational capacity at each resolution before the "
          "spatial size changes.")

pdf.add_page()

# ---- Full architecture ----
h1(pdf, "4. Full Architecture")
body(pdf, "Four downsampling levels, a bottleneck, and four upsampling levels with skip "
          "connections. Channel width is controlled by one parameter, base=16.")

h2(pdf, "4.1 Encoder (contracting path)")
code(pdf,
     'self.enc1 = conv_block(3, base)          # base = 16\n'
     'self.enc2 = conv_block(base, base * 2)\n'
     'self.enc3 = conv_block(base * 2, base * 4)\n'
     'self.enc4 = conv_block(base * 4, base * 8)\n'
     'self.pool = nn.MaxPool2d(2)')
body(pdf,
     "Each encoder block doubles the channel count (16, 32, 64, 128) -- the standard convention "
     "of doubling channels each time spatial resolution is halved. self.pool is a single 2x2 "
     "max-pooling layer (no learnable parameters), reused at every downsampling step: it keeps "
     "the maximum value in each non-overlapping 2x2 pixel block, halving height and width.")

h2(pdf, "4.2 Bottleneck")
code(pdf, 'self.bottleneck = conv_block(base * 8, base * 16)')
body(pdf, "The bottom of the U: one more block, 128 to 256 channels, at the smallest spatial "
          "resolution -- the most channels, least spatial detail, most abstract representation "
          "in the network.")

h2(pdf, "4.3 Decoder (expanding path)")
code(pdf,
     'self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)\n'
     'self.dec4 = conv_block(base * 16, base * 8)\n'
     'self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)\n'
     'self.dec3 = conv_block(base * 8, base * 4)\n'
     'self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)\n'
     'self.dec2 = conv_block(base * 4, base * 2)\n'
     'self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)\n'
     'self.dec1 = conv_block(base * 2, base)')
body(pdf,
     "Each up-step has two parts. First, a ConvTranspose2d (kernel 2, stride 2): a transposed "
     "convolution that doubles spatial size and halves channel count. Unlike simple nearest- or "
     "bilinear-upsampling, this has learnable weights -- the network learns how to upsample "
     "rather than following a fixed rule. Second, the conv_block's declared input channel count "
     "is double what the upsample alone would produce (e.g. dec4 takes base*16=256 in, not "
     "base*8=128) -- because the decoder concatenates the upsampled features with the matching-"
     "resolution encoder output (the skip connection) before the block runs, doubling the "
     "channel count going in.")

h2(pdf, "4.4 Output layer")
code(pdf, 'self.out = nn.Conv2d(base, 1, 1)')
body(pdf,
     "A final 1x1 convolution, 16 channels to 1. A 1x1 kernel looks at no neighboring pixels -- "
     "it is a per-pixel linear combination of that pixel's 16 feature values, collapsing them "
     "into the single output logit described in Section 2.")

pdf.add_page()

# ---- Forward pass / shapes table ----
h1(pdf, "5. Data Flow and Verified Tensor Shapes")
body(pdf,
     "The forward() method below is the actual, complete implementation (model.py, lines 38-49). "
     "Shapes shown are for a batch size of 1 with a 512x512 input -- the size actually used "
     "throughout this project -- and were confirmed by running the model, not calculated by hand.")
code(pdf,
     'def forward(self, x):\n'
     '    e1 = self.enc1(x)\n'
     '    e2 = self.enc2(self.pool(e1))\n'
     '    e3 = self.enc3(self.pool(e2))\n'
     '    e4 = self.enc4(self.pool(e3))\n'
     '    b = self.bottleneck(self.pool(e4))\n'
     '\n'
     '    d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))\n'
     '    d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))\n'
     '    d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))\n'
     '    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))\n'
     '    return self.out(d1)')

table_rows = [
    ("Stage", "Operation", "Output shape (N=1)"),
    ("Input", "--", "(1, 3, 512, 512)"),
    ("enc1", "conv_block(3->16)", "(1, 16, 512, 512)"),
    ("", "maxpool 2x2", "(1, 16, 256, 256)"),
    ("enc2", "conv_block(16->32)", "(1, 32, 256, 256)"),
    ("", "maxpool 2x2", "(1, 32, 128, 128)"),
    ("enc3", "conv_block(32->64)", "(1, 64, 128, 128)"),
    ("", "maxpool 2x2", "(1, 64, 64, 64)"),
    ("enc4", "conv_block(64->128)", "(1, 128, 64, 64)"),
    ("", "maxpool 2x2", "(1, 128, 32, 32)"),
    ("bottleneck", "conv_block(128->256)", "(1, 256, 32, 32)"),
    ("up4+concat(e4)", "transposed conv + skip", "(1, 256, 64, 64)"),
    ("dec4", "conv_block(256->128)", "(1, 128, 64, 64)"),
    ("up3+concat(e3)", "transposed conv + skip", "(1, 128, 128, 128)"),
    ("dec3", "conv_block(128->64)", "(1, 64, 128, 128)"),
    ("up2+concat(e2)", "transposed conv + skip", "(1, 64, 256, 256)"),
    ("dec2", "conv_block(64->32)", "(1, 32, 256, 256)"),
    ("up1+concat(e1)", "transposed conv + skip", "(1, 32, 512, 512)"),
    ("dec1", "conv_block(32->16)", "(1, 16, 512, 512)"),
    ("out", "1x1 conv (16->1)", "(1, 1, 512, 512)"),
]

pdf.ln(2)
col_w = [45, 70, 65]
pdf.set_font("Helvetica", "B", 9.5)
pdf.set_fill_color(230, 230, 230)
for w_, txt in zip(col_w, table_rows[0]):
    pdf.cell(w_, 7, txt, border=1, fill=True)
pdf.ln()
pdf.set_font("Courier", "", 8.5)
for row in table_rows[1:]:
    for w_, txt in zip(col_w, row):
        pdf.cell(w_, 6.2, txt, border=1)
    pdf.ln()

pdf.add_page()

# ---- Parameter counts ----
h1(pdf, "6. Parameter Count (verified by execution)")
body(pdf,
     "Computed by instantiating the model and summing p.numel() for every parameter tensor -- "
     "an exact count, not an estimate.")

param_rows = [
    ("Component", "Parameters"),
    ("enc1", "2,832"),
    ("enc2", "14,016"),
    ("enc3", "55,680"),
    ("enc4", "221,952"),
    ("bottleneck", "886,272"),
    ("up4", "131,200"),
    ("dec4", "443,136"),
    ("up3", "32,832"),
    ("dec3", "110,976"),
    ("up2", "8,224"),
    ("dec2", "27,840"),
    ("up1", "2,064"),
    ("dec1", "7,008"),
    ("out", "17"),
    ("TOTAL", "1,944,049"),
]
pdf.ln(2)
col_w2 = [90, 60]
pdf.set_font("Helvetica", "B", 9.5)
pdf.set_fill_color(230, 230, 230)
for w_, txt in zip(col_w2, param_rows[0]):
    pdf.cell(w_, 7, txt, border=1, fill=True)
pdf.ln()
pdf.set_font("Courier", "", 8.5)
for row in param_rows[1:]:
    is_total = row[0] == "TOTAL"
    if is_total:
        pdf.set_font("Courier", "B", 8.5)
    for w_, txt in zip(col_w2, row):
        pdf.cell(w_, 6.2, txt, border=1)
    pdf.ln()
    if is_total:
        pdf.set_font("Courier", "", 8.5)

pdf.ln(3)
body(pdf,
     "All ~1.94 million parameters are trainable; none are frozen or pretrained -- the model is "
     "trained entirely from scratch (see model_provenance section below). As a cross-check: the "
     "saved checkpoint file is 7,830,233 bytes, and 1,944,049 parameters x 4 bytes/float32 is "
     "approximately 7.78 MB, consistent with a plain float32 state dict, which is what "
     "torch.save(model.state_dict(), ...) produces.")
body(pdf,
     "For scale reference: this is a small model. A full-size U-Net on natural images often runs "
     "tens of millions of parameters; a ResNet-50 backbone alone is roughly 25 million.")

# ---- Design rationale ----
h1(pdf, "7. Design Choices and Their Justification")
body(pdf, "Being direct about which choices are principled versus which are reasonable but "
          "untested judgment calls:")
bullet(pdf, "base=16, 4 levels: chosen because the training set is small (32 to 107 images over "
             "the course of this project). A wider/deeper network has more parameters to overfit "
             "with, and is also slower to train on CPU -- the only hardware available in this "
             "environment. This was NOT tuned via systematic search; it is a reasonable starting "
             "guess, not a validated optimum.")
bullet(pdf, "4 downsampling levels: 512 / 2^4 = 32, a reasonably small but non-degenerate "
             "bottleneck. Not derived from a more principled calculation.")
bullet(pdf, "3x3 kernels with same-padding: the near-universal default in modern convolutional "
             "networks since VGG-style architectures -- a stack of 3x3 convolutions approximates "
             "a larger receptive field with fewer parameters than one large kernel.")
bullet(pdf, "ConvTranspose2d for upsampling (rather than fixed interpolation + convolution): a "
             "design choice, not benchmarked against the alternative in this project. Some "
             "practitioners prefer plain upsampling because transposed convolutions can produce "
             "checkerboard artifacts; both options were not compared here.")
bullet(pdf, "No activation on the final output: deliberate, so raw logits can be passed directly "
             "to a numerically-stable combined sigmoid+loss function during training.")

# ---- What this document does not cover ----
h1(pdf, "8. Scope of This Document")
body(pdf,
     "This document covers only the architecture defined in model.py. It does not cover: how the "
     "model is trained (loss function, optimizer, data augmentation, epochs -- defined in a "
     "separate training script), how its output is converted into a final centerline "
     "(confidence-weighted extraction and gap-bridging logic in separate inference code), what "
     "data it was trained on, or how that data was collected. Those are documented separately.")

pdf.output(OUT_PATH)
print(f"Saved: {OUT_PATH}")
