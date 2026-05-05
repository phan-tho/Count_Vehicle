"""
phase1_perspective.py
---------------------
Phase 1 of the IoT Vehicle Counting Pipeline.

Detects the 4 corner markers of the traffic simulation board and
applies a perspective transform to produce a flat top-down view.

Detection strategy (quadrant-based, most robust):
  • Divides the image into 4 quadrants (TL, TR, BR, BL).
  • In each quadrant independently, searches for the best square-shaped
    contour (allowing different sizes per quadrant).
  • This works even when the markers differ in physical size or the
    ArUco pattern is not in a standard dictionary.

Fallback: if the primary method fails, tries full-image ArUco detection.

Output: flattened_test.jpg (1280 × 720)
"""

import sys
import os
import numpy as np
import cv2

# ── Output dimensions ──────────────────────────────────────────────────────────
OUTPUT_W = 1280
OUTPUT_H = 720

# ── File paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH  = os.path.join(SCRIPT_DIR, "sa_ban_traffic.mp4")
SAMPLE_PATH = os.path.join(SCRIPT_DIR, "sample.png")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "flattened_test.jpg")


# ══════════════════════════════════════════════════════════════════════════════
#  Frame loading
# ══════════════════════════════════════════════════════════════════════════════

def load_best_frame() -> tuple:
    """
    Return (frame, source_name).
    Prefers sample.png (highest resolution); falls back to the video.
    """
    frame = cv2.imread(SAMPLE_PATH)
    if frame is not None:
        print(f"[INFO] Using sample.png: {frame.shape[1]}×{frame.shape[0]}")
        return frame, "sample"

    cap = cv2.VideoCapture(VIDEO_PATH)
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        if ok:
            print(f"[INFO] Using video frame: {frame.shape[1]}×{frame.shape[0]}")
            return frame, "video"

    raise FileNotFoundError(
        "Could not load sample.png or the video file.\n"
        f"  sample : {SAMPLE_PATH}\n  video  : {VIDEO_PATH}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Core detection utilities
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _find_square_candidates(region: np.ndarray,
                             min_area_frac: float = 0.001,
                             max_area_frac: float = 0.50) -> list:
    """
    Return a list of {'area', 'center', 'approx'} dicts for every
    roughly-square 4-sided contour found in *region*.

    min_area_frac / max_area_frac are fractions of the region's total area.
    """
    gray  = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray  = _preprocess(gray)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 80)

    kernel = np.ones((5, 5), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    rh, rw  = region.shape[:2]
    tot     = rw * rh
    min_a   = tot * min_area_frac
    max_a   = tot * max_area_frac

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_a <= area <= max_a):
            continue

        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4:
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        aspect = bw / max(bh, 1)
        if not (0.35 < aspect < 2.8):
            continue

        cx, cy = x + bw / 2.0, y + bh / 2.0
        results.append({
            "area":   area,
            "center": np.array([cx, cy], dtype="float32"),
            "approx": approx,
        })

    # Deduplicate (keep the larger when two are very close)
    results.sort(key=lambda c: c["area"], reverse=True)
    kept = []
    for c in results:
        if all(np.linalg.norm(c["center"] - k["center"]) > 20 for k in kept):
            kept.append(c)

    return kept


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 – Per-quadrant square detection (PRIMARY)
# ══════════════════════════════════════════════════════════════════════════════

def try_quadrant_detection(frame: np.ndarray):
    """
    Divide the image into 4 overlapping quadrants and find the best
    square-shaped marker candidate in each.

    Returns (ordered_pts, "quadrant") where ordered_pts is float32
    [TL, TR, BR, BL], or None on failure.
    """
    h, w = frame.shape[:2]
    # 55% overlap so markers near the true corners are fully inside a quadrant
    hx, hy = int(w * 0.55), int(h * 0.55)

    quadrant_defs = {
        "TL": (0,      0,      hx,    hy   ),
        "TR": (w - hx, 0,      w,     hy   ),
        "BR": (w - hx, h - hy, w,     h    ),
        "BL": (0,      h - hy, hx,    h    ),
    }

    corners = {}
    for label, (x1, y1, x2, y2) in quadrant_defs.items():
        region = frame[y1:y2, x1:x2]
        cands  = _find_square_candidates(region)

        if not cands:
            print(f"  [{label}] No candidate found in quadrant.")
            continue

        # Pick the candidate closest to the extreme corner of the quadrant
        corner_x = x1 if label in ("TL", "BL") else x2
        corner_y = y1 if label in ("TL", "TR") else y2

        best = None
        best_score = float("inf")
        for c in cands:
            # Absolute position in full image
            abs_cx = c["center"][0] + x1
            abs_cy = c["center"][1] + y1
            # Score = distance to the extreme corner, biased by 1/area
            dist  = np.hypot(abs_cx - corner_x, abs_cy - corner_y)
            score = dist  # prefer closer to the true corner
            if score < best_score:
                best_score = score
                best = {"center": np.array([abs_cx, abs_cy], dtype="float32"),
                        "area":   c["area"]}

        if best is not None:
            print(f"  [{label}] Best: center=({best['center'][0]:.0f},"
                  f"{best['center'][1]:.0f}), area={best['area']:.0f}")
            corners[label] = best["center"]

    if len(corners) < 4:
        print(f"[Quadrant] Only {len(corners)}/4 corners found.")
        return None

    ordered = np.array(
        [corners["TL"], corners["TR"], corners["BR"], corners["BL"]],
        dtype="float32"
    )
    return ordered, "quadrant"


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 – Full-image ArUco detection (FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

def try_aruco_detection(frame: np.ndarray):
    """
    Try all standard ArUco dictionaries with relaxed parameters.
    Returns (ordered_pts, "aruco") or None.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = _preprocess(gray)
    h, w  = gray.shape

    dict_ids = [
        cv2.aruco.DICT_4X4_50,  cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_5X5_100, cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]

    def make_params(win_max, poly_acc, min_peri):
        p = cv2.aruco.DetectorParameters()
        p.adaptiveThreshWinSizeMin    = 3
        p.adaptiveThreshWinSizeMax    = win_max
        p.adaptiveThreshWinSizeStep   = 10
        p.minMarkerPerimeterRate      = min_peri
        p.maxMarkerPerimeterRate      = 10.0
        p.polygonalApproxAccuracyRate = poly_acc
        p.minCornerDistanceRate       = 0.005
        p.minDistanceToBorder         = 1
        return p

    param_sets = [
        make_params(53,  0.05, 0.02),
        make_params(103, 0.10, 0.01),
    ]

    use_new = hasattr(cv2.aruco, "ArucoDetector")

    best_n, best_corners, best_ids = 0, None, None

    for dict_id in dict_ids:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        for params in param_sets:
            try:
                if use_new:
                    det = cv2.aruco.ArucoDetector(aruco_dict, params)
                    corners, ids, _ = det.detectMarkers(gray)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        gray, cv2.aruco.Dictionary_get(dict_id),
                        parameters=cv2.aruco.DetectorParameters_create()
                    )
            except Exception:
                continue

            if ids is not None and len(ids) > best_n:
                best_n, best_corners, best_ids = len(ids), corners, ids

    if best_ids is None or best_n < 4:
        return None

    centres = np.array(
        [c.reshape(4, 2).mean(axis=0) for c in best_corners], dtype="float32"
    )

    # Deduplicate
    kept = []
    for p in centres:
        if all(np.linalg.norm(p - k) >= 30 for k in kept):
            kept.append(p)
    centres = np.array(kept, dtype="float32")

    if len(centres) < 4:
        return None

    ordered = _pick_corners_by_quadrant(centres)
    print(f"[ArUco] {best_n} raw detections → 4 corners:")
    for label, pt in zip(["TL", "TR", "BR", "BL"], ordered):
        print(f"    {label}: ({pt[0]:.0f}, {pt[1]:.0f})")
    return ordered, "aruco"


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pick_corners_by_quadrant(pts: np.ndarray) -> np.ndarray:
    """
    From N ≥ 4 points, pick one representative per quadrant
    (TL / TR / BR / BL) relative to the centroid.
    """
    cx, cy = pts.mean(axis=0)
    result  = []
    masks = [
        ("TL", (pts[:, 0] <= cx) & (pts[:, 1] <= cy)),
        ("TR", (pts[:, 0] >  cx) & (pts[:, 1] <= cy)),
        ("BR", (pts[:, 0] >  cx) & (pts[:, 1] >  cy)),
        ("BL", (pts[:, 0] <= cx) & (pts[:, 1] >  cy)),
    ]
    for label, mask in masks:
        group = pts[mask]
        if len(group) == 0:
            # Fill with the extreme point from the whole set
            if label == "TL": result.append(pts[pts.sum(axis=1).argmin()])
            elif label == "BR": result.append(pts[pts.sum(axis=1).argmax()])
            elif label == "TR":
                d = pts[:, 0] - pts[:, 1]
                result.append(pts[d.argmin()])
            else:
                d = pts[:, 0] - pts[:, 1]
                result.append(pts[d.argmax()])
        else:
            if label == "TL": result.append(group[group.sum(axis=1).argmin()])
            elif label == "BR": result.append(group[group.sum(axis=1).argmax()])
            elif label == "TR":
                d = group[:, 0] - group[:, 1]
                result.append(group[d.argmin()])
            else:
                d = group[:, 0] - group[:, 1]
                result.append(group[d.argmax()])

    return np.array(result, dtype="float32")


def _is_valid_quad(ordered: np.ndarray, img_w: int, img_h: int) -> bool:
    """Reject degenerate or too-small quadrilaterals."""
    tl, tr, br, bl = ordered
    min_span = min(img_w, img_h) * 0.05
    for a, b in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
        if np.linalg.norm(a - b) < min_span:
            return False
    width  = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if width < img_w * 0.10 or height < img_h * 0.10:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load frame ──────────────────────────────────────────────────────────
    frame, source = load_best_frame()
    h, w = frame.shape[:2]

    # Save debug raw frame (scaled to ≤1000px for easy viewing)
    raw_debug = os.path.join(SCRIPT_DIR, "debug_raw_frame.jpg")
    scale_d = min(1.0, 1000 / max(w, h))
    cv2.imwrite(raw_debug,
                frame if scale_d == 1.0
                else cv2.resize(frame, (int(w * scale_d), int(h * scale_d))))
    print(f"[INFO] Raw frame saved → {raw_debug}")

    # ── 2. Detect the 4 corner markers ────────────────────────────────────────
    result = None

    print("[INFO] Strategy 1: per-quadrant square detection …")
    result = try_quadrant_detection(frame)

    if result is None or not _is_valid_quad(result[0], w, h):
        print("[INFO] Strategy 1 failed – trying ArUco detection …")
        result = try_aruco_detection(frame)

    if result is None or not _is_valid_quad(result[0], w, h):
        print(
            "\n[ERROR] Could not reliably locate the 4 board-corner markers.\n"
            "  Suggestions:\n"
            "    • Ensure all 4 ArUco markers are fully in the camera frame.\n"
            "    • Improve lighting (avoid glare / heavy shadows).\n"
            "    • Increase marker size if they appear tiny in the image.\n"
            f"\n  Inspect: {raw_debug}\n"
        )
        sys.exit(1)

    ordered, method = result
    labels = ["TL", "TR", "BR", "BL"]
    print(f"\n[INFO] Detection method: {method}")
    for label, pt in zip(labels, ordered):
        print(f"       {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

    # ── 3. Debug overlay ───────────────────────────────────────────────────────
    debug_frame = frame.copy()
    scale_disp  = min(1.0, 1200 / max(w, h))
    dot_r = max(8, int(18 / scale_disp))
    colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0)]

    for label, pt, color in zip(labels, ordered, colors):
        cv2.circle(debug_frame, tuple(pt.astype(int)), dot_r, color, -1)
        cv2.putText(
            debug_frame, label,
            (int(pt[0]) + dot_r + 4, int(pt[1]) - dot_r),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.8, 1.8 / scale_disp), (255, 255, 0),
            max(2, int(3 / scale_disp)),
        )
    cv2.polylines(
        debug_frame,
        [ordered.astype(int).reshape((-1, 1, 2))],
        isClosed=True, color=(0, 255, 255),
        thickness=max(2, int(4 / scale_disp)),
    )

    debug_markers = os.path.join(SCRIPT_DIR, "debug_markers.jpg")
    disp = (debug_frame if scale_disp == 1.0
            else cv2.resize(debug_frame,
                            (int(w * scale_disp), int(h * scale_disp))))
    cv2.imwrite(debug_markers, disp)
    print(f"[INFO] Marker debug image saved → {debug_markers}")

    # ── 4. Perspective transform ───────────────────────────────────────────────
    dst_pts = np.array(
        [
            [0,            0           ],   # TL
            [OUTPUT_W - 1, 0           ],   # TR
            [OUTPUT_W - 1, OUTPUT_H - 1],   # BR
            [0,            OUTPUT_H - 1],   # BL
        ],
        dtype="float32",
    )

    M      = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(frame, M, (OUTPUT_W, OUTPUT_H))

    # ── 5. Save output ─────────────────────────────────────────────────────────
    cv2.imwrite(OUTPUT_PATH, warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n[SUCCESS] Flattened image saved → {OUTPUT_PATH}")
    print(f"          Output size: {OUTPUT_W} × {OUTPUT_H} px")
    print(
        "\nManual verification:\n"
        "  ✓ Board boundaries should be perfectly straight (no trapezoid).\n"
        "  ✓ Background / camera frame fully cropped out.\n"
        "  ✓ The view looks like a perfect top-down 2-D rectangle.\n"
    )


if __name__ == "__main__":
    main()
