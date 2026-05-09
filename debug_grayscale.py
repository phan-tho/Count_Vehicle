import cv2
import numpy as np
import json
import os

INPUT_IMAGE_PATH = 'sample_img/phase2/phase2_masked_lanes.png'
ROI_JSON_PATH = 'lane_rois.json'
DEBUG_DIR = 'sample_img/debug_gray_rois'

def clean_mask(mask, k_size=5):
    kernel = np.ones((k_size, k_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def count_contours(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return len(valid_contours)

def process_roi_percentile(gray):
    # Method 1: Percentile based thresholding
    med_gray = np.median(gray)
    max_gray = np.percentile(gray, 99) # Top 1% to ignore tiny noise spikes
    
    # If the max brightness is very close to median, it's likely an empty lane.
    if max_gray - med_gray < 30:
        return np.zeros_like(gray, dtype=np.uint8)
        
    thresh = med_gray + (max_gray - med_gray) * 0.4
    mask = (gray > thresh).astype(np.uint8) * 255
    return mask

def process_roi_otsu(gray):
    # Method 2: Otsu thresholding but with a contrast check
    min_gray = np.min(gray)
    max_gray = np.max(gray)
    
    # If contrast is too low, empty lane
    if max_gray - min_gray < 40:
        return np.zeros_like(gray, dtype=np.uint8)
        
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu might pick a very low threshold if it's mostly dark
    if ret < med_gray(gray) + 20:
        return np.zeros_like(gray, dtype=np.uint8)
    return mask

def med_gray(gray):
    return np.median(gray)

def process_roi_adaptive(gray):
    # Method 3: Adaptive Gaussian Thresholding
    # Since background is dark, vehicles are bright. 
    # C is subtracted from mean, so pixel > mean - C -> we need pixel > mean + margin -> C should be negative
    # Vehicles are blocks, so block size should be large
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 51, -20)
                                 
    # To avoid noise in empty lanes, mask it by the overall max brightness
    if np.max(gray) - np.median(gray) < 30:
        return np.zeros_like(gray, dtype=np.uint8)
    return mask

def process_roi_fixed(gray):
    # Method 4: Fixed thresholding assuming vehicles are always brighter than say 120
    # and background is < 80.
    ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    return mask

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
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        m_perc = process_roi_percentile(gray)
        m_otsu = process_roi_otsu(gray)
        m_adap = process_roi_adaptive(gray)
        m_fix = process_roi_fixed(gray)
        
        # Clean
        m_perc = clean_mask(m_perc)
        m_otsu = clean_mask(m_otsu)
        m_adap = clean_mask(m_adap)
        m_fix = clean_mask(m_fix)
        
        c_perc = count_contours(m_perc)
        c_otsu = count_contours(m_otsu)
        c_adap = count_contours(m_adap)
        c_fix = count_contours(m_fix)
        
        std_gray = np.std(gray)
        print(f"ROI {i:02d}: Otsu={c_otsu} | Std={std_gray:.2f} | Max={np.max(gray)}")
        
        # Visualize
        def c3(m): return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        
        top_row = np.hstack([roi_bgr, c3(gray), c3(m_perc)])
        bot_row = np.hstack([c3(m_otsu), c3(m_adap), c3(m_fix)])
        debug_img = np.vstack([top_row, bot_row])
        
        h_d, w_d = debug_img.shape[:2]
        debug_img = cv2.resize(debug_img, (w_d * 2, h_d * 2))
        
        cv2.imwrite(os.path.join(DEBUG_DIR, f"roi_{i:02d}.png"), debug_img)

if __name__ == "__main__":
    main()
