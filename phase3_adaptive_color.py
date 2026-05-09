import cv2
import numpy as np
import json
import os

INPUT_IMAGE_PATH = 'sample_img/phase2/phase2_masked_lanes.png'
ROI_JSON_PATH = 'lane_rois.json'
OUTPUT_VIZ_PATH = 'sample_img/phase3_adaptive_output.png'

def process_roi_adaptive(roi_bgr):
    # Convert to float for mathematical operations
    bgr_float = roi_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(bgr_float)

    # 1. Excess Red (ExR) and Excess Green (ExG) indices
    # These indices highlight the specific colors relative to others, mitigating lighting shifts.
    ExR = 1.4 * R - G - B
    ExG = 2.0 * G - R - B

    # 2. Adaptive Thresholding using Mean and Standard Deviation per ROI
    # For Red
    mean_ExR = np.mean(ExR)
    std_ExR = np.std(ExR)
    # Lowered k from 2.5 to 1.5 to not omit vehicles, but increased min_thresh slightly to avoid background noise
    k_red = 1.5
    min_thresh_red = 0.12 
    thresh_red = max(min_thresh_red, mean_ExR + k_red * std_ExR)
    
    red_mask = (ExR > thresh_red).astype(np.uint8) * 255

    # For Green
    mean_ExG = np.mean(ExG)
    std_ExG = np.std(ExG)
    k_green = 1.5
    min_thresh_green = 0.12 
    thresh_green = max(min_thresh_green, mean_ExG + k_green * std_ExG)
    
    green_mask = (ExG > thresh_green).astype(np.uint8) * 255
    
    # 3. Morphological operations to clean up noise (increased kernel size for robustness)
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    return red_mask, green_mask

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

    # We will also create a combined mask image for debugging
    full_red_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_green_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for (x, y, w, h) in rois:
        # Extract ROI
        roi_bgr = img[y:y+h, x:x+w]
        
        # Apply adaptive color segmentation
        red_mask, green_mask = process_roi_adaptive(roi_bgr)
        
        # Place masks back into the full size mask
        full_red_mask[y:y+h, x:x+w] = red_mask
        full_green_mask[y:y+h, x:x+w] = green_mask
        
        # Draw bounding boxes around detected vehicles on the visualization image
        # Find Red Vehicles
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > 100: # Filter small noise (increased from 50 to 100)
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                # Draw on full image (offset by ROI coordinates)
                cv2.rectangle(viz_img, (x+rx, y+ry), (x+rx+rw, y+ry+rh), (0, 0, 255), 2)
                
        # Find Green Vehicles
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_green:
            area = cv2.contourArea(cnt)
            if area > 100: # Filter small noise (increased from 50 to 100)
                gx, gy, gw, gh = cv2.boundingRect(cnt)
                cv2.rectangle(viz_img, (x+gx, y+gy), (x+gx+gw, y+gy+gh), (0, 255, 0), 2)

        # Draw ROI boundaries
        cv2.rectangle(viz_img, (x, y), (x+w, y+h), (255, 255, 255), 1)

    os.makedirs(os.path.dirname(OUTPUT_VIZ_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_VIZ_PATH, viz_img)
    cv2.imwrite('sample_img/phase3_red_mask.png', full_red_mask)
    cv2.imwrite('sample_img/phase3_green_mask.png', full_green_mask)
    
    print(f"Processed {len(rois)} ROIs.")
    print(f"Saved visualizations to {OUTPUT_VIZ_PATH}")

if __name__ == "__main__":
    main()
