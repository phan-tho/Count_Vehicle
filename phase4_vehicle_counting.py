import cv2
import json
import os
import numpy as np

def main():
    mask_path = 'sample_img/phase3/phase3_grayscale_mask.png'
    vis_path = 'sample_img/phase2/phase2_masked_lanes.png'
    lane_rois_path = 'lane_rois.json'
    vehicle_params_path = 'vehicle_params.json'
    output_dir = 'sample_img/phase4'
    output_path = os.path.join(output_dir, 'phase4_counting_result.png')

    # 1. Load images
    if not os.path.exists(mask_path):
        print(f"Error: Mask image not found at {mask_path}")
        return
    if not os.path.exists(vis_path):
        print(f"Error: Visualization image not found at {vis_path}")
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    vis_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)

    # 2. Read lane ROIs
    if not os.path.exists(lane_rois_path):
        print(f"Error: Lane ROIs file not found at {lane_rois_path}")
        return
    with open(lane_rois_path, 'r') as f:
        lane_rois = json.load(f)

    # 3. Read vehicle params
    if not os.path.exists(vehicle_params_path):
        print(f"Error: Vehicle params file not found at {vehicle_params_path}")
        return
    with open(vehicle_params_path, 'r') as f:
        params = json.load(f)
    
    min_area = params['MIN_VEHICLE_AREA']
    avg_area = params['AVERAGE_VEHICLE_AREA']

    print(f"Loaded params: MIN_VEHICLE_AREA={min_area}, AVERAGE_VEHICLE_AREA={avg_area}")

    # 4. Initialize counters
    total_vehicles = 0
    lane_counts = []

    # 5. Iterate through each ROI
    for i, roi in enumerate(lane_rois):
        x, y, w, h = roi
        lane_vehicle_count = 0

        # Draw ROI bounding box (White)
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Crop the mask
        cropped_mask = mask[y:y+h, x:x+w]

        # Find contours
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Ignore noise
            if area < min_area:
                continue
            
            # Calculate estimated vehicles
            estimated_vehicles = max(1, int(round(area / avg_area)))
            lane_vehicle_count += estimated_vehicles

            # 6. Draw contour bounding box and label
            cx, cy, cw, ch = cv2.boundingRect(contour)
            abs_x = x + cx
            abs_y = y + cy

            # Draw bounding box around contour (Cyan)
            cv2.rectangle(vis_img, (abs_x, abs_y), (abs_x + cw, abs_y + ch), (255, 255, 0), 2)

            # Draw text label (estimated vehicles)
            cv2.putText(vis_img, f"x{estimated_vehicles}", (abs_x, max(abs_y - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Add lane total to overall total
        total_vehicles += lane_vehicle_count
        lane_counts.append(lane_vehicle_count)

        # Draw lane total at the top of the lane ROI
        cv2.putText(vis_img, f"Lane {i+1}: {lane_vehicle_count} veh", (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"Lane {i+1} count: {lane_vehicle_count}")

    print(f"\nTotal vehicles across all lanes: {total_vehicles}")

    # 7. Create output dir and save
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"Saved visualization result to {output_path}")

if __name__ == '__main__':
    main()
