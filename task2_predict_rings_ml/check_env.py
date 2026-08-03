# check_env.py
# ----------------------------------------------------------------------
# Run this FIRST. It checks every package is installed. If it prints
# "ALL GOOD", you're ready. If something fails, re-run:
#     pip install -r requirements.txt
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# ----------------------------------------------------------------------

import sys                                           # built-in: shows the Python version

print("Python version:", sys.version)
print("-" * 60)

ok = True                                            # becomes False if any check fails

# Each item: (nice name, what to import).
checks = [
    ("ultralytics", "ultralytics"),
    ("OpenCV (cv2)", "cv2"),
    ("NumPy", "numpy"),
    ("pandas", "pandas"),
    ("Matplotlib", "matplotlib"),
    ("SciPy", "scipy"),
]

for nice_name, module_name in checks:
    try:
        module = __import__(module_name)             # try to load the package
        version = getattr(module, "__version__", "?")  # read its version if it has one
        print(f"OK   {nice_name:14s} version {version}")
    except Exception as error:                       # not installed -> report it
        ok = False
        print(f"FAIL {nice_name:14s} -> {error}")

print("-" * 60)
if ok:
    print("ALL GOOD. You're ready. Next: run 01_segment_pick.py")
else:
    print("Some packages are missing. Run:  pip install -r requirements.txt")
