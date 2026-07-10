# check_env.py
# ----------------------------------------------------------------------
# Run this file FIRST, before the numbered scripts.
# It checks that every package is installed correctly. If it prints
# "ALL GOOD", your environment is ready. If it complains about a missing
# package, re-run the install step in SETUP.md.
#
# HOW TO RUN IN PYCHARM: open this file and press the green Run button.
# -------------
# ---------------------------------------------------------

import sys   # built-in: lets us print the Python version

# Show which Python is being used (useful when something is wrong).
print("Python version:", sys.version)
print("-" * 60)

# We will try to import each package one by one and report the result.
# A "package" is a toolbox of code; "import" means "load that toolbox".
ok = True   # we flip this to False if anything fails

# Each item is: (name to show, what to import, attribute that holds the version)
checks = [
    ("OpenCV (cv2)", "cv2", "__version__"),
    ("NumPy",         "numpy", "__version__"),
    ("pandas",        "pandas", "__version__"),
    ("Matplotlib",    "matplotlib", "__version__"),
    ("SciPy",         "scipy", "__version__"),
    ("openpyxl",      "openpyxl", "__version__"),
]

for nice_name, module_name, version_attr in checks:
    try:
        module = __import__(module_name)              # try to load the toolbox
        version = getattr(module, version_attr, "?")  # read its version number
        print(f"OK   {nice_name:14s} version {version}")
    except Exception as error:                        # if it is not installed
        ok = False                                    # remember that something failed
        print(f"FAIL {nice_name:14s} -> {error}")

print("-" * 60)

# Final verdict.
if ok:
    print("ALL GOOD. Your environment is ready. You can run 01_load_image.py next.")
else:
    print("Some packages are missing. Open SETUP.md and re-do the install step:")
    print("    pip install -r requirements.txt")
