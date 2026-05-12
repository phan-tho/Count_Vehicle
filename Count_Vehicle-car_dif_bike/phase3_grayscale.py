import cv2
import numpy as np
import json
import os

INPUT_IMAGE_PATH = 'sample_img/phase2/phase2_masked_lanes.png'
ROI_JSON_PATH = 'lane_rois.json'
OUTPUT_VIZ_PATH = 'sample_img/phase3/phase3_grayscale_output.png'
OUTPUT_MASK_PATH = 'sample_img/phase3/phase3_grayscale_mask.png'

def process_roi_grayscale(gray):
    """
    Process a single ROI using grayscale Otsu thresholding.
    Returns a binary mask of detected vehicles.
    """
    # 1. Empty Lane Check
    # If the standard deviation of brightness is very low, or the maximum brightness 
    # doesn't reach the typical brightness of a painted vehicle, the lane is empty.
    if np.std(gray) < 10 or np.max(gray) < 120:
        return np.zeros_like(gray, dtype=np.uint8)

    # 2. Otsu's Thresholding
    # Otsu automatically finds the optimal threshold to separate the bright vehicles 
    # from the darker background lane.
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Safety check: if Otsu picked a threshold that is too low (e.g. splitting noise)
    if ret < np.median(gray) + 15:
        return np.zeros_like(gray, dtype=np.uint8)

    # 3. Morphological Cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def main():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: {INPUT_IMAGE_PATH} not found.")
        return
        
    if not os.path.exists(ROI_JSON_PATH):
        print(f"Error: {ROI_JSON_PATH} not found.")
        return

    img = cv2.imread(INPUT_IMAGE_PATH)
    viz_img = img.copy()

    with open(ROI_JSON_PATH, 'r') as f:
        rois = json.load(f)

    # Create a full-size combined mask for debugging
    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for i, (x, y, w, h) in enumerate(rois):
        roi_bgr = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply grayscale thresholding
        mask = process_roi_grayscale(gray)
        
        # Place the ROI mask back into the full size mask
        full_mask[y:y+h, x:x+w] = mask
        
        # Highlight ROI boundaries on visualization image
        cv2.rectangle(viz_img, (x, y), (x+w, y+h), (255, 255, 255), 1)

    os.makedirs(os.path.dirname(OUTPUT_VIZ_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_VIZ_PATH, viz_img)
    cv2.imwrite(OUTPUT_MASK_PATH, full_mask)
    
    print(f"Processed {len(rois)} ROIs.")
    print(f"Saved visualization to {OUTPUT_VIZ_PATH}")

if __name__ == "__main__":
    main()
