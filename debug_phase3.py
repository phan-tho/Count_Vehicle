import cv2
import numpy as np
import json
import os

INPUT_IMAGE_PATH = 'sample_img/phase2/phase2_masked_lanes.png'
ROI_JSON_PATH = 'lane_rois.json'
DEBUG_DIR = 'sample_img/debug_rois'

def process_roi_exr(roi_bgr, k_std=2.0, min_thresh=0.08):
    bgr_float = roi_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(bgr_float)
    ExR = 1.4 * R - G - B
    ExG = 2.0 * G - R - B

    # For Red
    mean_ExR = np.mean(ExR)
    std_ExR = np.std(ExR)
    thresh_red = max(min_thresh, mean_ExR + k_std * std_ExR)
    red_mask = (ExR > thresh_red).astype(np.uint8) * 255

    # For Green
    mean_ExG = np.mean(ExG)
    std_ExG = np.std(ExG)
    thresh_green = max(min_thresh, mean_ExG + k_std * std_ExG)
    green_mask = (ExG > thresh_green).astype(np.uint8) * 255
    
    return red_mask, green_mask

def process_roi_hsv(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    # Adaptive threshold for S and V based on ROI median
    # We expect vehicles to have higher saturation and reasonable value
    # median_s = np.median(S)
    # median_v = np.median(V)
    # min_s = max(40, median_s * 0.8)
    # min_v = max(40, median_v * 0.5)
    
    # Actually, a fixed, wide threshold for S and V often works better if H is well-bounded.
    # Because vehicles are painted, S > 50, V > 50. Let's make it very loose to catch everything
    # and let the color hue do the heavy lifting.
    
    # Red hue wraps around: 0-15 and 165-180
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Green hue: ~40-90
    lower_green = np.array([40, 100, 80])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    return red_mask, green_mask

def process_roi_otsu(roi_bgr):
    bgr_float = roi_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(bgr_float)
    ExR = 1.4 * R - G - B
    ExG = 2.0 * G - R - B
    
    # Convert ExR and ExG to 8-bit for Otsu
    # ExR and ExG can be negative. Let's clip to [0, max] and normalize to [0, 255]
    ExR_clipped = np.clip(ExR, 0, None)
    ExG_clipped = np.clip(ExG, 0, None)
    
    # Only normalize if max is large enough to contain a vehicle
    max_ExR = np.max(ExR_clipped)
    max_ExG = np.max(ExG_clipped)
    
    if max_ExR > 0.05:
        ExR_8u = (ExR_clipped / max_ExR * 255).astype(np.uint8)
        ret, red_mask = cv2.threshold(ExR_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu can be too generous. Let's verify the threshold corresponds to an ExR > 0.05
        real_thresh = (ret / 255.0) * max_ExR
        if real_thresh < 0.05:
            red_mask = (ExR > 0.05).astype(np.uint8) * 255
    else:
        red_mask = np.zeros_like(R, dtype=np.uint8)
        
    if max_ExG > 0.05:
        ExG_8u = (ExG_clipped / max_ExG * 255).astype(np.uint8)
        ret, green_mask = cv2.threshold(ExG_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        real_thresh = (ret / 255.0) * max_ExG
        if real_thresh < 0.05:
            green_mask = (ExG > 0.05).astype(np.uint8) * 255
    else:
        green_mask = np.zeros_like(G, dtype=np.uint8)

    return red_mask, green_mask


def process_roi_percentile(roi_bgr):
    bgr_float = roi_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(bgr_float)
    ExR = 1.4 * R - G - B
    ExG = 2.0 * G - R - B
    
    # Red
    med_R = np.median(ExR)
    max_R = np.percentile(ExR, 99) # Top 1% to ignore tiny noise spikes
    thresh_red = max(0.08, med_R + (max_R - med_R) * 0.4)
    red_mask = (ExR > thresh_red).astype(np.uint8) * 255

    # Green
    med_G = np.median(ExG)
    max_G = np.percentile(ExG, 99)
    thresh_green = max(0.08, med_G + (max_G - med_G) * 0.4)
    green_mask = (ExG > thresh_green).astype(np.uint8) * 255
                                       
    return red_mask, green_mask
def clean_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def count_contours(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return len(valid_contours)

def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    img = cv2.imread(INPUT_IMAGE_PATH)
    if img is None:
        print("Failed to load image")
        return
        
    with open(ROI_JSON_PATH, 'r') as f:
        rois = json.load(f)

    for i, (x, y, w, h) in enumerate(rois):
        roi_bgr = img[y:y+h, x:x+w]
        
        # Test 1: HSV
        r_hsv, g_hsv = process_roi_hsv(roi_bgr)
        r_hsv, g_hsv = clean_mask(r_hsv), clean_mask(g_hsv)
        
        # Test 2: Percentile
        r_otsu, g_otsu = process_roi_percentile(roi_bgr)
        r_otsu, g_otsu = clean_mask(r_otsu), clean_mask(g_otsu)
        
        # Test 3: ExR with relaxed thresholds
        r_exr, g_exr = process_roi_exr(roi_bgr, k_std=1.2, min_thresh=0.08)
        r_exr, g_exr = clean_mask(r_exr), clean_mask(g_exr)
        
        c_hsv = (count_contours(r_hsv), count_contours(g_hsv))
        c_otsu = (count_contours(r_otsu), count_contours(g_otsu))
        c_exr = (count_contours(r_exr), count_contours(g_exr))
        
        print(f"ROI {i:02d}: HSV=(R:{c_hsv[0]},G:{c_hsv[1]}) | Otsu=(R:{c_otsu[0]},G:{c_otsu[1]}) | ExR_relax=(R:{c_exr[0]},G:{c_exr[1]})")
        
        # Create a side-by-side debug image
        # Top row: BGR, HSV Red, Otsu Red, ExR_relax Red
        # Bot row: empty, HSV Grn, Otsu Grn, ExR_relax Grn
        
        # Convert masks to 3-channel for stacking
        def c3(m): return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        
        top_row = np.hstack([roi_bgr, c3(r_hsv), c3(r_otsu), c3(r_exr)])
        empty = np.zeros_like(roi_bgr)
        bot_row = np.hstack([empty, c3(g_hsv), c3(g_otsu), c3(g_exr)])
        
        debug_img = np.vstack([top_row, bot_row])
        
        # Resize if it's too small to see
        # Let's scale by 2x just to make it readable
        h_d, w_d = debug_img.shape[:2]
        debug_img = cv2.resize(debug_img, (w_d * 2, h_d * 2))
        
        cv2.imwrite(os.path.join(DEBUG_DIR, f"roi_{i:02d}.png"), debug_img)
        
if __name__ == "__main__":
    main()
