import cv2
import numpy as np
import json
import os

# Define paths
INPUT_IMAGE_PATH = 'sample_img/phase1/phase1_output.png'
OUTPUT_JSON_PATH = 'lane_rois.json'
OUTPUT_MASKED_IMAGE_PATH = 'phase2_masked_lanes.png'

def main():
    # 1. Load the image
    # Assuming the phase1_output is in sample_img/phase1/ based on previous commands
    # If it's in the root folder, adjust the path accordingly.
    if not os.path.exists(INPUT_IMAGE_PATH):
        # Fallback to current directory just in case
        INPUT_IMAGE_PATH_FALLBACK = 'phase1_output.png'
        if os.path.exists(INPUT_IMAGE_PATH_FALLBACK):
            print(f"Found image at {INPUT_IMAGE_PATH_FALLBACK}")
            img = cv2.imread(INPUT_IMAGE_PATH_FALLBACK)
        else:
            print(f"Error: Could not find image at {INPUT_IMAGE_PATH} or {INPUT_IMAGE_PATH_FALLBACK}")
            return
    else:
        img = cv2.imread(INPUT_IMAGE_PATH)

    if img is None:
        print("Error: Could not load the image.")
        return

    # 2. Print instructions for the user
    print("=" * 60)
    print("INSTRUCTIONS FOR ROI SELECTION:")
    print("1. Drag your mouse to draw a bounding box over a traffic lane.")
    print("2. Press ENTER or SPACE to confirm the current selection.")
    print("3. Repeat steps 1 and 2 for all lanes you want to include.")
    print("4. Press ESC when you are finished selecting all lanes.")
    print("=" * 60)

    # OpenCV window name
    window_name = "Select Lane ROIs (Press ESC to finish)"
    
    # Create a resizable window and set a reasonable initial size
    # This allows the image to fit on your screen, but the returned coordinates 
    # will still be relative to the original image size.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # We set it to 800x800 as a default viewing size, but you can resize the window further
    cv2.resizeWindow(window_name, 800, 800)
    
    # cv2.selectROIs allows multiple selections.
    # It returns a list of bounding boxes: (x, y, w, h)
    rois = cv2.selectROIs(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if len(rois) == 0:
        print("No ROIs were selected. Exiting without saving.")
        return

    # 3. Save coordinates to JSON
    # Convert numpy arrays to standard python lists for JSON serialization
    rois_list = [int(v) for roi in rois for v in roi]
    # Restructure into list of lists
    rois_list = [list(roi) for roi in rois]
    # Make sure elements are standard int, not numpy types
    rois_list = [[int(val) for val in roi] for roi in rois_list]
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(rois_list, f, indent=4)
    print(f"Successfully saved {len(rois_list)} ROIs to {OUTPUT_JSON_PATH}")

    # 4. Create a black mask of the same dimensions (height, width)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for (x, y, w, h) in rois:
        # Draw a filled white rectangle for each selected ROI on the mask
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

    # 5. Apply the mask to the original image using bitwise_and
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 6. Save the resulting masked image
    cv2.imwrite(OUTPUT_MASKED_IMAGE_PATH, masked_img)
    print(f"Successfully saved masked image to {OUTPUT_MASKED_IMAGE_PATH}")

if __name__ == "__main__":
    main()
