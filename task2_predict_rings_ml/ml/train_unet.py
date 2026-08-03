"""
Train the small U-Net on the auto-labeled data from auto_label.py.
No manual labeling involved -- labels are FastSAM pseudo-masks.

Run:
    ./.venv/bin/python ml/train_unet.py
"""
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import SmallUNet

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "data", "images")
MASKS_DIR = os.path.join(HERE, "data", "masks")
CKPT_PATH = os.path.join(HERE, "shell_unet.pt")

SIZE = 512
VAL_FRACTION = 0.15
EPOCHS = 150
BATCH_SIZE = 4
LR = 1e-3
SEED = 0


class ShellDataset(Dataset):
    def __init__(self, tags, augment):
        self.tags = tags
        self.augment = augment

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, i):
        tag = self.tags[i]
        img = cv2.imread(os.path.join(IMAGES_DIR, tag + ".png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS_DIR, tag + ".png"), cv2.IMREAD_GRAYSCALE)

        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
            k = random.choice([0, 1, 2, 3])
            if k:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            if random.random() < 0.5:
                gain = random.uniform(0.8, 1.2)
                bias = random.uniform(-20, 20)
                img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tags = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.endswith(".png"))
    random.shuffle(tags)
    n_val = max(1, int(len(tags) * VAL_FRACTION))
    val_tags, train_tags = tags[:n_val], tags[n_val:]
    print(f"{len(train_tags)} train / {len(val_tags)} val images")

    train_ds = ShellDataset(train_tags, augment=True)
    val_ds = ShellDataset(val_tags, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = SmallUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # the target is a thin band (~3-4% of pixels), so an unweighted BCE loss
    # lets the model minimize loss by mostly predicting background -- causing
    # the "only predicts a stub" behavior seen without any weight at all.
    # But the full inverse-frequency weight (~28x) overcorrected: it taught
    # the model to claim broad, plausible-looking "this could be shell"
    # territory rather than the precise band (false negatives got penalized
    # so heavily that over-predicting became the safe bet), producing thick,
    # lopsided blobs offset from the true centerline. Use sqrt of the
    # inverse-frequency weight instead -- a standard softening for this exact
    # failure mode -- to balance recall against precision instead of
    # maximizing recall alone.
    fg_fracs = []
    for tag in train_tags:
        m = cv2.imread(os.path.join(MASKS_DIR, tag + ".png"), cv2.IMREAD_GRAYSCALE)
        fg_fracs.append((m > 127).mean())
    fg_frac = max(np.mean(fg_fracs), 1e-4)
    pos_weight = torch.tensor([((1 - fg_frac) / fg_frac) ** 0.5], device=device)
    print(f"foreground fraction {fg_frac:.4f} -> pos_weight {pos_weight.item():.1f}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y) + dice_loss(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_PATH)
            marker = "  (saved)"
        print(f"epoch {epoch:3d}/{EPOCHS}  train {train_loss:.4f}  val {val_loss:.4f}{marker}")

    print(f"\nbest val loss {best_val:.4f}, checkpoint at {CKPT_PATH}")


if __name__ == "__main__":
    main()
