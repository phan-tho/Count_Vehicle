import cv2
import numpy as np
import json
import os

def main():
    # File paths
    MASK_PATH = 'sample_img/phase3/phase3_grayscale_mask.png'
    BGR_PATH = 'sample_img/phase2/phase2_masked_lanes.png'
    ROIS_PATH = 'lane_rois.json'
    PARAMS_PATH = 'vehicle_params.json'
    OUT_DIR = 'sample_img/phase4'
    OUT_PATH = os.path.join(OUT_DIR, 'phase4_counting_result.png')

    # Create output directory if it doesn't exist
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Load images
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask from {MASK_PATH}")
        return

    bgr_img = cv2.imread(BGR_PATH)
    if bgr_img is None:
        print(f"Error: Could not load BGR image from {BGR_PATH}")
        return

    output_img = bgr_img.copy()

    # Load parameters
    try:
        with open(ROIS_PATH, 'r') as f:
            rois = json.load(f)
    except Exception as e:
        print(f"Error loading {ROIS_PATH}: {e}")
        return

    try:
        with open(PARAMS_PATH, 'r') as f:
            params = json.load(f)
    except Exception as e:
        print(f"Error loading {PARAMS_PATH}: {e}")
        return

    MIN_VEHICLE_AREA = params.get("MIN_VEHICLE_AREA", 200)
    AVERAGE_CAR_AREA = params.get("AVERAGE_CAR_AREA", 895)
    AVERAGE_BIKE_AREA = params.get("AVERAGE_BIKE_AREA", 400)

    print("--- Phase 4: Vehicle Counting ---")
    print(f"Parameters: MIN_VEHICLE_AREA={MIN_VEHICLE_AREA}, CAR_AREA={AVERAGE_CAR_AREA}, BIKE_AREA={AVERAGE_BIKE_AREA}")

    total_cars = 0
    total_bikes = 0

    # Process each ROI (lane)
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        
        roi_mask = mask[y:y+h, x:x+w]
        roi_bgr = bgr_img[y:y+h, x:x+w]
        
        # Find contours
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lane_cars = 0
        lane_bikes = 0
        
        # Draw lane boundaries (Blue rectangle)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_VEHICLE_AREA:
                continue
                
            # Create a blank single-contour mask
            single_mask = np.zeros(roi_mask.shape, dtype=np.uint8)
            cv2.drawContours(single_mask, [cnt], -1, 255, -1)
            
            # Extract pixels using int16 to prevent underflow
            b = roi_bgr[:, :, 0].astype(np.int16)
            g = roi_bgr[:, :, 1].astype(np.int16)
            r = roi_bgr[:, :, 2].astype(np.int16)
            
            in_mask = single_mask > 0
            
            # Count Red pixels: R > G + 20 and R > B + 20
            red_cond = (r > g + 20) & (r > b + 20) & in_mask
            red_px = np.sum(red_cond)
            
            # Count Green pixels: G > R + 20 and G > B + 20
            green_cond = (g > r + 20) & (g > b + 20) & in_mask
            green_px = np.sum(green_cond)
            
            total_color_px = red_px + green_px
            if total_color_px == 0:
                continue
                
            # Calculate proportions
            r_ratio = red_px / total_color_px
            g_ratio = green_px / total_color_px
            
            # Estimate counts based on area and proportions
            cars = max(0, round((area * r_ratio) / AVERAGE_CAR_AREA))
            bikes = max(0, round((area * g_ratio) / AVERAGE_BIKE_AREA))
            
            lane_cars += cars
            lane_bikes += bikes
            
            # Draw bounding box for the contour
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            global_x = x + cx
            global_y = y + cy
            
            cv2.rectangle(output_img, (global_x, global_y), (global_x + cw, global_y + ch), (0, 255, 255), 2)
            
            # Add text label for specific counts
            label = f"C:{cars} B:{bikes}"
            text_y = global_y - 5 if global_y - 5 > 10 else global_y + 15
            cv2.putText(output_img, label, (global_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
        # Draw text label above the lane
        lane_label = f"Lane {i}: Cars={lane_cars} Bikes={lane_bikes}"
        lane_text_y = y - 10 if y - 10 > 20 else y + h + 20
        cv2.putText(output_img, lane_label, (x, lane_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Accumulate and print details
        print(f"Lane {i}: Cars = {lane_cars}, Bikes = {lane_bikes}")
        total_cars += lane_cars
        total_bikes += lane_bikes

    print("---------------------------------")
    print(f"Total Counts - Cars: {total_cars}, Bikes: {total_bikes}")
    
    cv2.imwrite(OUT_PATH, output_img)
    print(f"Saved result to: {OUT_PATH}")

if __name__ == "__main__":
    main()
