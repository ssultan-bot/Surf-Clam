# Setup (do this once)

You need a normal computer. No fancy graphics card needed.

## 1. Install Python
Go to <https://www.python.org/downloads/> and click the big download button.
Open the file. On the first screen, **check the box "Add Python to PATH"**, then
click **Install Now**.

## 2. Install PyCharm
Go to <https://www.jetbrains.com/pycharm/download/> and get **PyCharm Community
Edition** (the free one). Install it and open it.

## 3. Open the project
In PyCharm: **Open** -> pick the **`task1_alignlabels`** folder. If it asks
anything, just click **OK**.

## 4. Create the environment and install the toolboxes
At the bottom of PyCharm, click **Terminal**. Type these lines, one at a time,
pressing Ecnter after each:

```
cd D:\MyWork\Code\SurfClam
py -m venv surf
surf\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ **The path in line 1 is just an example.** Replace
> `D:\MyWork\Code\SurfClam` with the folder where YOUR project actually is.
> (Tip: in Windows Explorer, hold **Shift**, right-click the folder, choose
> **Copy as path**, and paste it.)

- Line 1 moves into your project folder.
- Line 2 makes a clean, private environment (a folder named `surf`).
- Line 3 turns it on. You'll see `(surf)` appear at the start of the line.
- Line 4 installs all the toolboxes. Wait for it to finish.

> If `py` doesn't work, try `python` instead. On Mac, use `python3`, and line 3
> is `source surf/bin/activate`.

**If line 3 gives a red error saying "running scripts is disabled on this
system":** that's just PowerShell blocking scripts for safety. Run this one line
first (it only affects the current window), type `Y` and Enter if it asks, then
run line 3 again:

```4
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 5. Check it worked
Open **`check_env.py`** and press the green **Run** button.
If you see **"ALL GOOD"**, you're ready.

## 6. Start
Open **`01_load_image.py`** and press **Run**. Then try 02, 03, 04, 05.

---

**If a script can't find the photo:** open `config.py` and fix the two file paths
at the top so they point to your image and Excel file.
