import cv2
import numpy as np
import os

def main():
    image_path = 'sample_img/phase1/phase1_output.png'
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find {image_path}")
        return

    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load {image_path}")
        return
        
    print(f"Successfully loaded {image_path}")

    # 2. Phase 2 (Pre-processing): Apply Gaussian Blur
    # A 5x5 kernel is chosen to subtly blur the image, removing high-frequency noise 
    # like moiré patterns from a screen without losing the shape of the vehicles.
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Phase 3 (Color Segmentation): Convert from BGR to HSV
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    cv2.imwrite('sample_img/phase2-3/hsv_img.png', hsv_img)

    # 4. Define the HSV lower and upper bounds
    # Hue ranges:
    # Red: wraps around 0-10 and 160-180 in OpenCV (which uses 0-179 for Hue)
    # Green: typically around 40-90 in OpenCV
    # Saturation and Value are set to catch most bright/dark variations.
    
    # Red thresholds
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Green thresholds
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    # 5. Create binary masks
    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_red_combined = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # 6. Apply morphological operations
    # MORPH_OPEN removes small noise (false positives) in the background.
    # MORPH_CLOSE fills small holes (false negatives) inside the detected vehicles.
    kernel = np.ones((5, 5), np.uint8)
    
    # Clean Red Mask
    mask_red_clean = cv2.morphologyEx(mask_red_combined, cv2.MORPH_OPEN, kernel)
    mask_red_clean = cv2.morphologyEx(mask_red_clean, cv2.MORPH_CLOSE, kernel)
    
    # Clean Green Mask
    mask_green_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green_clean = cv2.morphologyEx(mask_green_clean, cv2.MORPH_CLOSE, kernel)

    # 7. Save the final cleaned masks
    cv2.imwrite('sample_img/phase2-3/mask_red.png', mask_red_clean)
    cv2.imwrite('sample_img/phase2-3/mask_green.png', mask_green_clean)
    print("Saved mask_red.png and mask_green.png")

    # 8. Create a combined visualization
    # Combine the red and green masks
    combined_mask = cv2.bitwise_or(mask_red_clean, mask_green_clean)
    
    # Bitwise AND to keep only the segmented regions in their original colors
    segmented_vehicles = cv2.bitwise_and(img, img, mask=combined_mask)
    
    cv2.imwrite('sample_img/phase2-3/segmented_vehicles.png', segmented_vehicles)
    print("Saved segmented_vehicles.png")

if __name__ == '__main__':
    main()
