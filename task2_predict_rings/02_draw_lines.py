# 02_draw_lines.py
# ----------------------------------------------------------------------
# STEP 2: Automatically trace the clam hinge/tail curve using the shell mask.
#
# This script starts from an interior hinge region and follows a skeleton path
# outward toward the shell edge. It prefers interior ridge segments and
# rejects the outer boundary perimeter.
# ----------------------------------------------------------------------
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

import config


def clamp_point(pt, shape):
    x, y = pt
    return (max(0, min(int(x), shape[1] - 1)), max(0, min(int(y), shape[0] - 1)))


def compute_column_boundaries(mask, x_min=None, x_max=None, min_pixels=10):
    """Return per-column top and bottom boundaries as dicts {x: (top,bottom)}."""
    h, w = mask.shape
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = w - 1
    tops = {}
    bottoms = {}
    for x in range(max(0, x_min), min(w, x_max + 1)):
        rows = np.where(mask[:, x])[0]
        if rows.size < min_pixels:
            continue
        tops[x] = int(rows.min())
        bottoms[x] = int(rows.max())
    return tops, bottoms


def compute_interior_growth_curve(mask, hinge_seed, tail_seed, top_frac=0.30, min_pixels=10, smooth_sigma=12, x_min=None, x_max=None):
    """Compute interior growth curve as y = top + top_frac*(bottom-top) per x.
    Returns (path list, tops dict, bottoms dict). By default spans full mask width
    unless x_min/x_max are provided.
    """
    h, w = mask.shape
    if x_min is None:
        x0 = 0
    else:
        x0 = int(max(0, x_min))
    if x_max is None:
        x1 = w - 1
    else:
        x1 = int(min(w - 1, x_max))
    tops, bottoms = compute_column_boundaries(mask, x0, x1, min_pixels=min_pixels)
    xs = []
    ys = []
    # Use only columns that have valid top/bottom entries
    xs_available = [x for x in range(x0, x1 + 1) if x in tops]
    if len(xs_available) == 0:
        return [], tops, bottoms
    xmin_av = xs_available[0]
    xmax_av = xs_available[-1]
    denom = float(xmax_av - xmin_av) if xmax_av != xmin_av else 1.0
    for x in xs_available:
        top = tops[x]
        bottom = bottoms[x]
        # t increases from 0 at left to 1 at right; fraction increases toward hinge (right)
        t = float(x - xmin_av) / denom
        fraction = float(top_frac) + 0.15 * t
        y = top + fraction * (bottom - top)
        xs.append(x)
        ys.append(y)
    if len(xs) < 3:
        return [], tops, bottoms
    ys_smooth = gaussian_filter1d(np.array(ys, dtype=float), sigma=smooth_sigma)
    try:
        spline = UnivariateSpline(xs, ys_smooth, k=3, s=0.0)
        xs_full = np.arange(xs[0], xs[-1] + 1)
        ys_full = spline(xs_full)
        path = [(int(x), int(round(y))) for x, y in zip(xs_full, ys_full)]
        return path, tops, bottoms
    except Exception:
        from scipy.interpolate import interp1d

        f = interp1d(xs, ys_smooth, kind="linear")
        xs_new = np.arange(xs[0], xs[-1] + 1)
        ys_new = f(xs_new)
        path = [(int(x), int(round(y))) for x, y in zip(xs_new, ys_new)]
        return path, tops, bottoms


def map_point_to_image(point, img_shape, mask_shape):
    x, y = point
    img_h, img_w = img_shape[:2]
    mask_h, mask_w = mask_shape
    if (img_h, img_w) == (mask_h, mask_w):
        return (x, y)
    sx = img_w / mask_w
    sy = img_h / mask_h
    return (int(round(x * sx)), int(round(y * sy)))


def map_path_to_image(path, img_shape, mask_shape):
    img_h, img_w = img_shape[:2]
    mask_h, mask_w = mask_shape
    if (img_h, img_w) == (mask_h, mask_w):
        return path, 1.0, 1.0
    sx = img_w / mask_w
    sy = img_h / mask_h
    mapped = [(int(round(x * sx)), int(round(y * sy))) for x, y in path]
    return mapped, sx, sy


def log_coordinate_debug(img, mask, path, seed, start, end, sx, sy):
    img_h, img_w = img.shape[:2]
    mask_h, mask_w = mask.shape
    print("--- COORDINATE DEBUG ---")
    print(f"Image dims: {img_w}x{img_h}")
    print(f"Mask dims: {mask_w}x{mask_h}")
    print(f"Mask-original dims match: {(img_h, img_w) == (mask_h, mask_w)}")
    print(f"Scale factors: sx={sx:.4f}, sy={sy:.4f}")
    print(f"Path length: {len(path)}")
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    print(f"Path x range: {min(xs)}..{max(xs)}")
    print(f"Path y range: {min(ys)}..{max(ys)}")
    print(f"Seed: {seed}, Start: {start}, End: {end}")
    if not (0 <= seed[0] < img_w and 0 <= seed[1] < img_h):
        print("WARNING: seed is outside image bounds")
    if not (0 <= start[0] < img_w and 0 <= start[1] < img_h):
        print("WARNING: start is outside image bounds")
    if not (0 <= end[0] < img_w and 0 <= end[1] < img_h):
        print("WARNING: end is outside image bounds")
    outside = [p for p in path if not (0 <= p[0] < img_w and 0 <= p[1] < img_h)]
    print(f"Path points outside image bounds: {len(outside)}")
    if len(outside) > 0:
        print("Sample out-of-bounds points:", outside[:10])
    if mask_h == img_h and mask_w == img_w:
        points_in_mask = [mask[p[1], p[0]] if 0 <= p[0] < mask_w and 0 <= p[1] < mask_h else False for p in path]
        print(f"Path points inside mask: {sum(points_in_mask)}/{len(path)}")
    print("------------------------")


def compute_mask_edges(mask):
    mask_uint = (mask.astype(np.uint8) * 255)
    edges = cv2.Canny(mask_uint, 50, 150)
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return edges, contours


def save_debug_visuals(img, mask, edges, tops, bottoms, path, seed, start, end, boundary_candidates=None, path_dist_stats=None):
    mask_overlay = img.copy()
    mask_overlay[mask] = (
        mask_overlay[mask].astype(np.float32) * 0.5 + np.array([0, 0, 255], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    cv2.imwrite("debug_mask_overlay.png", mask_overlay)

    edge_vis = img.copy()
    edge_vis[edges > 0] = (0, 255, 255)
    cv2.imwrite("debug_edge_pixels.png", edge_vis)

    def draw_roi(image, p1, p2, padding=30):
        x0, y0 = p1
        x1, y1 = p2
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(image.shape[1] - 1, x1 + padding)
        y1 = min(image.shape[0] - 1, y1 + padding)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 255), 1)

    sk_vis = img.copy()
    # draw upper (blue) and lower (yellow) boundaries if provided
    if tops is not None and len(tops) > 0:
        upper_pts = np.array([(x, tops[x]) for x in sorted(tops.keys())], dtype=np.int32)
        if len(upper_pts) > 1:
            cv2.polylines(sk_vis, [upper_pts], False, (255, 0, 0), 2)
    if bottoms is not None and len(bottoms) > 0:
        lower_pts = np.array([(x, bottoms[x]) for x in sorted(bottoms.keys())], dtype=np.int32)
        if len(lower_pts) > 1:
            cv2.polylines(sk_vis, [lower_pts], False, (0, 255, 255), 2)
    if len(path) > 0:
        cv2.polylines(sk_vis, [np.array(path, dtype=np.int32)], False, (0, 0, 255), 2)
    if boundary_candidates is not None:
        for pt in boundary_candidates:
            cv2.circle(sk_vis, pt, 4, (0, 255, 255), -1)
    cv2.circle(sk_vis, seed, 6, (0, 0, 255), -1)
    cv2.circle(sk_vis, start, 6, (255, 255, 0), -1)
    cv2.circle(sk_vis, end, 6, (255, 0, 0), -1)
    draw_roi(sk_vis, seed, end)
    cv2.imwrite("debug_skeleton_path.png", sk_vis)

    path_on_mask = np.zeros_like(img)
    path_on_mask[mask] = (255, 255, 255)
    # draw upper/lower boundaries on path mask
    if tops is not None and len(tops) > 0:
        upper_pts = np.array([(x, tops[x]) for x in sorted(tops.keys())], dtype=np.int32)
        if len(upper_pts) > 1:
            cv2.polylines(path_on_mask, [upper_pts], False, (255, 0, 0), 2)
    if bottoms is not None and len(bottoms) > 0:
        lower_pts = np.array([(x, bottoms[x]) for x in sorted(bottoms.keys())], dtype=np.int32)
        if len(lower_pts) > 1:
            cv2.polylines(path_on_mask, [lower_pts], False, (0, 255, 255), 2)
    if len(path) > 0:
        cv2.polylines(path_on_mask, [np.array(path, dtype=np.int32)], False, (0, 255, 0), 3)
    if boundary_candidates is not None:
        for pt in boundary_candidates:
            cv2.circle(path_on_mask, pt, 4, (0, 255, 255), -1)
    cv2.circle(path_on_mask, seed, 6, (0, 0, 255), -1)
    cv2.circle(path_on_mask, start, 6, (255, 255, 0), -1)
    cv2.circle(path_on_mask, end, 6, (255, 0, 0), -1)
    if path_dist_stats is not None:
        txt = f"dist min={path_dist_stats[0]:.1f} mean={path_dist_stats[1]:.1f} max={path_dist_stats[2]:.1f}"
        cv2.putText(path_on_mask, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    draw_roi(path_on_mask, seed, end)
    cv2.imwrite("debug_path_before_overlay.png", path_on_mask)

    print("Saved debug_mask_overlay.png, debug_edge_pixels.png, debug_skeleton_path.png, debug_path_before_overlay.png")


def trace_hinge_path(mask):
    if config.HINGE_SEED is None:
        raise ValueError("config.HINGE_SEED must be set to a hinge coordinate in config.py.")
    hinge_seed = clamp_point(config.HINGE_SEED, mask.shape)
    if not mask[hinge_seed[1], hinge_seed[0]]:
        raise ValueError("config.HINGE_SEED must be inside the shell mask.")

    if config.TAIL_SEED is None:
        raise ValueError("config.TAIL_SEED must be set to a tail coordinate in config.py.")
    tail_seed = clamp_point(config.TAIL_SEED, mask.shape)
    if not mask[tail_seed[1], tail_seed[0]]:
        raise ValueError("config.TAIL_SEED must be inside the shell mask.")

    print(f"Manual seeds HINGE_SEED={config.HINGE_SEED}, TAIL_SEED={config.TAIL_SEED}")
    print(f"HINGE SEED: {hinge_seed}")
    print(f"TAIL SEED: {tail_seed}")

    # Diagnose column validity across full image width
    h, w = mask.shape
    x_min = 0
    x_max = w - 1
    tops_pre, bottoms_pre = compute_column_boundaries(mask, x_min, x_max, min_pixels=10)
    print("DEBUG: xmin", x_min, "xmax", x_max)
    print("DEBUG: HINGE_SEED", config.HINGE_SEED, "TAIL_SEED", config.TAIL_SEED)
    print("DEBUG: valid columns before filtering:", len(tops_pre))

    # Compute interior growth curve across full width (not restricted to seed x-range)
    path, tops, bottoms = compute_interior_growth_curve(mask, hinge_seed, tail_seed, top_frac=0.30, min_pixels=10, smooth_sigma=12, x_min=x_min, x_max=x_max)
    print("DEBUG: generated interior points after filtering:", len(path))
    if len(path) == 0:
        raise ValueError("Unable to compute interior growth curve between the provided seeds.")

    start = path[0]
    end = path[-1]
    path_dist_stats = (0.0, 0.0, 0.0)

    print(f"PATH LENGTH: {len(path)}")
    return path, hinge_seed, start, end, (tops, bottoms), path_dist_stats


if __name__ == "__main__":
    img = cv2.imread(config.IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {config.IMG_PATH}")

    if not os.path.exists(config.BEST_MASK_FILE):
        raise FileNotFoundError("No saved mask. Run 01_segment_pick.py first.")

    mask = np.load(config.BEST_MASK_FILE)
    if mask.dtype != bool:
        mask = mask > 0

    print("Generating new trace from mask; not loading existing lines.npy")
    path, seed, start, end, tb, path_dist_stats = trace_hinge_path(mask)
    tops, bottoms = tb
    path, sx, sy = map_path_to_image(path, img.shape, mask.shape)
    seed_img = map_point_to_image(seed, img.shape, mask.shape)
    start_img = map_point_to_image(start, img.shape, mask.shape)
    end_img = map_point_to_image(end, img.shape, mask.shape)
    np.save(config.LINES_FILE, np.array([path], dtype=object), allow_pickle=True)

    edges, contours = compute_mask_edges(mask)
    save_debug_visuals(img, mask, edges, tops, bottoms, path, seed_img, start_img, end_img, boundary_candidates=None, path_dist_stats=path_dist_stats)

    log_coordinate_debug(img, mask, path, seed_img, start_img, end_img, sx, sy)

    overlay = img.copy()
    overlay[mask > 0] = (
        overlay[mask > 0].astype(np.float32) * 0.55 + np.array([0, 180, 255], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    display = overlay.copy()

    cv2.circle(display, seed_img, 8, (0, 0, 255), -1)
    cv2.putText(display, "SEED", (seed_img[0] + 8, seed_img[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.circle(display, start_img, 8, (255, 255, 0), -1)
    cv2.putText(display, "START", (start_img[0] + 8, start_img[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.circle(display, end_img, 8, (255, 0, 0), -1)
    cv2.putText(display, "END", (end_img[0] + 8, end_img[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    debug_mask = np.zeros_like(display)
    debug_mask[mask] = (0, 255, 0)
    display = cv2.addWeighted(display, 0.8, debug_mask, 0.2, 0)

    path_arr = np.array(path, dtype=np.int32)
    print("First 10 path points:", path_arr[:10].tolist())
    print("Path array shape:", path_arr.shape)
    print("Path x range:", path_arr[:, 0].min(), path_arr[:, 0].max())
    print("Path y range:", path_arr[:, 1].min(), path_arr[:, 1].max())
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)

    cv2.polylines(display, [path_arr], False, (0, 255, 100), 3)
    if len(path_arr) > 0:
        cv2.circle(display, tuple(path_arr[0]), 10, (0, 0, 255), -1)
        cv2.putText(display, "TRACE START", (path_arr[0][0] + 12, path_arr[0][1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Automatic hinge/tail curve", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("automatic_hinge_curve.png", display)
    print(f"Saved automatic hinge/tail curve to: {config.LINES_FILE}")
    print("Saved automatic_hinge_curve.png")
