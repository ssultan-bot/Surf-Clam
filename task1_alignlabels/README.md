# Align Labels on a Clam Hinge Photo

This project puts a clam's yearly growth measurements (from an Excel file)
onto a photo of its hinge.

**Setup:** see `SETUP.md` first (install Python + the packages). Do it once.

## The files

Each script can be opened in PyCharm and run on its own (press the green Run
button). Nothing is saved to disk — each one just shows a window.

- `config.py` — settings. Put your photo path and Excel path here. Not run by itself.
- `01_load_image.py` — opens the photo and shows the numbers a computer sees.
- `02_read_labels.py` — reads the growth measurements from Excel and shows a bar chart.
- `03_draw_curve.py` — you trace the hinge with your mouse.
- `04_project_labels.py` — you click two points; it puts the real measurements on the line and prints them.
- `05_visualize.py` — same as 04, but makes the final two-panel picture.
- `run_all.py` — the full real version with a hand-drawn curve, start to finish.
- `check_env.py` — run this once to confirm the packages are installed.

## How to run

1. Open `config.py` and set the two file paths at the top to your photo and Excel.
2. Run `check_env.py`. If it says **ALL GOOD**, you're ready.
3. Run `01`, then `02`, `03`, `04`, `05` — or just run `run_all.py`.

In `04`, `05`, and `run_all`: a window opens — **click the TAIL, then the ROOT**
(for `03` and `run_all`, drag to trace the curve), then press **ENTER**.

## If something breaks

- **Can't find the photo** → fix the paths in `config.py`.
- **Missing package** (e.g. `No module named cv2`) → see `SETUP.md`.
