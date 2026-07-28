import os
import cv2, numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev


def segment(gray):
    g = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        boxes = stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]
        shell_like = np.where(areas / boxes < 0.9)[0]
        pool = shell_like if len(shell_like) else np.arange(len(areas))
        big = 1 + pool[np.argmax(areas[pool])]
        m = np.uint8(lab == big) * 255
    return ndi.binary_fill_holes(m > 0).astype(np.uint8) * 255


def bottom_edge_path(mask):
    xs = np.where(mask.any(axis=0))[0]
    ys = np.array([np.where(mask[:, x] > 0)[0].max() for x in xs], dtype=float)
    return np.column_stack([xs.astype(float), ys])


def centerline_path(mask, max_thickness_ratio=1.5, median_win=101):
    """For each column, the midpoint of the growth band.

    Naively bisecting the column's full top-to-bottom span breaks wherever
    debris or a broken fragment is fused onto the shell in that column: the
    mask is tall there for reasons that have nothing to do with the growth
    band, so the true midpoint is nowhere near the band. Clipping the
    thickness alone isn't enough either, because when material is fused
    directly onto the ventral side, y_bot itself is contaminated (it's no
    longer the shell's true edge) -- offsetting up from a wrong anchor just
    gives a different wrong answer. So instead: flag any column whose raw
    thickness balloons past the local (median-filtered) expectation as
    unreliable, throw its midpoint away entirely, and bridge the gap by
    interpolating between the nearest columns on either side that weren't
    flagged -- the same "don't trust it, bridge across it" approach already
    used for the near-root chip/break case."""
    xs = np.where(mask.any(axis=0))[0]
    y_top = np.empty(len(xs), dtype=float)
    y_bot = np.empty(len(xs), dtype=float)
    for i, x in enumerate(xs):
        col = np.where(mask[:, x] > 0)[0]
        y_top[i] = col.min()
        y_bot[i] = col.max()
    thickness = y_bot - y_top
    baseline = ndi.median_filter(thickness, size=median_win, mode="nearest")
    bad = thickness > baseline * max_thickness_ratio

    mids = y_bot - thickness / 2.0
    if bad.any() and not bad.all():
        idx = np.arange(len(xs))
        mids[bad] = np.interp(idx[bad], idx[~bad], mids[~bad])
    return np.column_stack([xs.astype(float), mids])


def despike(path, sigma=50, thresh=12.0, iters=6):
    n = len(path)
    y = path[:, 1].astype(float)
    x = np.arange(n, dtype=float) - n / 2
    weight = np.ones(n)
    for _ in range(iters):
        S0  = gaussian_filter1d(weight, sigma, mode="nearest")
        S1  = gaussian_filter1d(weight * x, sigma, mode="nearest")
        S2  = gaussian_filter1d(weight * x * x, sigma, mode="nearest")
        Sy  = gaussian_filter1d(weight * y, sigma, mode="nearest")
        Sxy = gaussian_filter1d(weight * x * y, sigma, mode="nearest")
        denom = S0 * S2 - S1 * S1
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        slope = (S0 * Sxy - S1 * Sy) / denom
        intercept = (Sy - slope * S1) / np.clip(S0, 1e-9, None)
        trend = intercept + slope * x
        weight = (np.abs(y - trend) <= thresh).astype(float)
    bad = weight == 0
    out = path.copy()
    if bad.any():
        idx = np.arange(n)
        out[bad, 1] = np.interp(idx[bad], idx[~bad], y[~bad])
    return out


def trim_near(path, xy):
    xy = np.asarray(xy, dtype=float)
    i = int(np.argmin(np.hypot(path[:, 0] - xy[0], path[:, 1] - xy[1])))
    if np.hypot(*(path[0] - xy)) < np.hypot(*(path[-1] - xy)):
        return path[i:]
    return path[:i + 1]


def relax_near_root(path, root_xy, gray, frac=0.10, extra_sigma=40, bulge_px=50):
    root_xy = np.asarray(root_xy, dtype=float)
    root_first = np.hypot(*(path[0] - root_xy)) <= np.hypot(*(path[-1] - root_xy))
    p = path if root_first else path[::-1]

    k = max(4, int(len(p) * frac))
    near = p[:k]
    nx = gaussian_filter1d(near[:, 0], sigma=extra_sigma, mode="nearest")
    ny = gaussian_filter1d(near[:, 1], sigma=extra_sigma, mode="nearest")
    w = np.linspace(1, 0, k)             # 1 at the root, fading to 0 by index k
    out = p.copy()
    out[:k, 0] = nx * w + near[:, 0] * (1 - w)
    out[:k, 1] = ny * w + near[:, 1] * (1 - w)

    chord = out[0] - out[k - 1]
    normal = np.array([-chord[1], chord[0]])
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    mid = out[k // 2]
    probe = 25
    h, w_img = gray.shape
    a = gray[int(np.clip(mid[1] + normal[1] * probe, 0, h - 1)),
             int(np.clip(mid[0] + normal[0] * probe, 0, w_img - 1))]
    b = gray[int(np.clip(mid[1] - normal[1] * probe, 0, h - 1)),
             int(np.clip(mid[0] - normal[0] * probe, 0, w_img - 1))]
    if b > a:
        normal = -normal
    bulge = np.sin(np.linspace(0, np.pi, k)) * bulge_px
    out[:k, 0] += normal[0] * bulge
    out[:k, 1] += normal[1] * bulge
    return out if root_first else out[::-1]


def smooth(path, n=600, median_size=9, sigma=90):
    x = ndi.median_filter(path[:, 0], size=median_size, mode="nearest")
    y = ndi.median_filter(path[:, 1], size=median_size, mode="nearest")
    x = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    y = gaussian_filter1d(y, sigma=sigma, mode="nearest")
    pts = np.column_stack([x, y])
    keep = np.r_[True, np.any(np.abs(np.diff(pts, axis=0)) > 1e-6, axis=1)]
    pts = pts[keep]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=3)
    xs, ys = splev(np.linspace(0, 1, n), tck)
    return np.column_stack([xs, ys])


def pick_root_click(img_bgr, win_w=1400, win_h=420):
    h, w = img_bgr.shape[:2]
    win_name = "Click the ROOT (hinge/umbo) point | R reset | ENTER confirm"
    pt_win = []

    def render():
        canvas = cv2.resize(img_bgr, (win_w, win_h))
        if pt_win:
            cv2.circle(canvas, pt_win[0], 7, (0, 0, 255), -1)
        cv2.putText(canvas, "Click ROOT (hinge/umbo) | R reset | ENTER confirm",
                    (20, win_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt_win[:] = [(x, y)]
            cv2.imshow(win_name, render())

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_w, win_h)
    cv2.setMouseCallback(win_name, on_mouse)
    cv2.imshow(win_name, render())
    while True:
        key = cv2.waitKey(20)
        if key == 13 and pt_win:             # ENTER (with a point placed) -> done
            break
        if key in (ord("r"), ord("R")):     # R -> reset
            pt_win.clear()
            cv2.imshow(win_name, render())
    cv2.destroyAllWindows()

    px, py = pt_win[0]
    return np.array([px * w / win_w, py * h / win_h])


def orient_root_first(path, root_xy):
    d0 = np.hypot(*(path[0]  - root_xy))
    d1 = np.hypot(*(path[-1] - root_xy))
    return path if d0 < d1 else path[::-1]


def anchor_to_root(path, root_xy, tol=3.0):
    """Make the line actually start at the clicked root, not just at whichever
    traced point happens to be closest to it. Bridges the small gap left when
    a chip/break near the hinge keeps the mask from reaching the exact root
    pixel (see METHODS.md); path[0] is assumed root-nearest (orient_root_first
    already called)."""
    root_xy = np.asarray(root_xy, dtype=float)
    if np.hypot(*(path[0] - root_xy)) <= tol:
        return path
    return np.vstack([root_xy, path])


def overlay(img_bgr, path):
    out = img_bgr.copy()
    pts = path.astype(np.int32)
    cv2.polylines(out, [pts], False, (255, 255, 0), 2)
    cv2.circle(out, tuple(pts[0]),  6, (0, 0, 255), -1)   # root, red
    cv2.circle(out, tuple(pts[-1]), 6, (0, 255, 0), -1)   # tail, green
    return out


# ---- run on one image ----
if __name__ == "__main__":
    import config

    img_path  = config.IMG_PATH

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask    = segment(gray)
    root_xy = pick_root_click(img, config.WIN_W, config.WIN_H)

    edge = centerline_path(mask)                          # trace through the middle of the shell
    edge = despike(edge)                                  # drop mid-body breaks/debris
    edge = trim_near(edge, root_xy)                      # stop at the clicked root, not past it
    edge = relax_near_root(edge, root_xy, gray)           # bow across the break, not around it
    path = smooth(edge, sigma=config.CENTERLINE_SMOOTH_SIGMA)
    path = orient_root_first(path, root_xy)
    path = anchor_to_root(path, root_xy)          # line must start exactly at the clicked root

    output_path = os.path.join(config.INTERMEDIATE_DIR, "labeled.png")
    cv2.imwrite(output_path, overlay(img, path))
    print(f"saved {output_path}")