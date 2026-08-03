# Task 2: Predict Growth Rings

This project finds a clam's growth rings **automatically** from the photo.
(Task 1 placed known measurements onto the photo; this one detects the rings
by itself.)

How it works in one line: an AI finds the shell, you draw a line along it, and
the program counts the **dark bands** (each dark band = one growth ring).

## Install (once)

You need Python (see Task 1's `SETUP.md` if you haven't installed it). Then, in
the PyCharm Terminal:

```
pip install -r requirements.txt
```

The first time you run Step 1, it downloads the FastSAM AI model, so you need
internet that one time. Everything runs on the **CPU — no graphics card needed**.

## The files

- `config.py` — settings: the photo path and the numbers you can tweak.
- `01_segment_pick.py` — the AI splits the photo into shapes; you click the shell.
- `02_draw_lines.py` — you draw line(s) along the growth direction.
- `03_grayscale.py` — reads color + gray along the line; compares color vs gray.
- `04_clahe.py` — boosts contrast (CLAHE); compares gray before vs after.
- `05_gaussian.py` — smooths the brightness curve; compares jagged vs smooth.
- `06_detect_and_save.py` — finds the rings, shows the result, saves CSV files.
- `run_all.py` — does all of the above in one run, showing only the final result.
- `check_env.py` — run once to confirm the packages are installed.

The steps pass data along through a small `intermediate/` folder, so run them
in order.

## How to run

1. Run `check_env.py`. If it says **ALL GOOD**, continue.
2. Run the six steps **in order**:

```
01_segment_pick  ->  02_draw_lines  ->  03_grayscale  ->  04_clahe  ->  05_gaussian  ->  06_detect_and_save
```

The first two open a window:
- **Step 1:** click the mask that is the shell, press **ENTER**.
- **Step 2:** drag to draw a line; right-click undoes; press **ENTER**.

Steps 3–6 each pop up a comparison chart — just look and close it. After step 6,
CSV files are saved next to the photo (`..._line1_pixels.csv` and
`..._line1_rings.csv`).

## Tweaking results

Open `config.py` and change these if the ring count looks wrong:

- `PROMINENCE` — smaller finds more (fainter) rings; bigger finds fewer.
- `MIN_DIST` — the smallest allowed gap between two rings.
- `GRAY_SIGMA` — more smoothing = fewer tiny false rings.
