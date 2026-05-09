# Vehicle Counting Pipeline for Adaptive Traffic Lights

## Project Objective
Develop a robust computer vision pipeline to count the number of vehicles (represented by red and green blocks) in each lane from a video recording of a traffic simulation board (`sa_ban_traffic.mp4`). This data will subsequently be used to feed an adaptive traffic light control algorithm.

## Pipeline Evaluation
The proposed pipeline is highly optimal and perfectly suited for this specific scenario. Here is why:
1. **ArUco Markers:** The presence of ArUco markers at the four corners of the board is a massive advantage. It allows for an exact, mathematically precise perspective transformation. This eliminates errors caused by camera angle or minor camera movements.
2. **Grayscale Otsu Thresholding per ROI:** Since Step 1 guarantees a fixed perspective, the physical coordinates of the lanes will never change. By defining Static ROIs early, we can crop out the lanes and eliminate 90% of background noise. Furthermore, converting ROIs to grayscale and applying Otsu's thresholding dynamically perfectly separates the bright vehicles from the darker background without struggling against dynamic illumination shifts and color casts.

---

## Phased Implementation Plan

### Phase 1: Perspective Transform & Alignment
*   **Goal:** Convert the angled camera view into a perfect, flat, top-down 2D view. Fix the coordinate system.
*   **Techniques:** 
    *   ArUco marker detection to find the 4 corners.
    *   Calculate the perspective transformation matrix.
    *   Warp the perspective to crop and flatten the board.
*   **Libraries:** `cv2` (OpenCV), `cv2.aruco`, `numpy`.

### Phase 2: Interactive Lane ROI Selection & Masking
*   **Goal:** Eliminate environmental noise by selecting static bounding boxes for the lanes and masking out the rest of the board.
*   **Techniques:** 
    *   Use an interactive selector (`cv2.selectROIs`) to define lane boundaries.
    *   Save these coordinates to a configuration file (`lane_rois.json`) for persistence.
    *   Apply a mask to black out everything outside the selected lanes.
*   **Libraries:** `cv2`, `json`.

### Phase 3: Grayscale Otsu Vehicle Masking per ROI
*   **Goal:** Mask the vehicles (bright blocks) in each lane regardless of their specific color, ensuring immunity to color casts and shadows.
*   **Techniques:**
    *   Convert each lane ROI independently into Grayscale.
    *   Use statistical checks (Standard Deviation) to safely ignore empty lanes.
    *   Apply Otsu's Thresholding to dynamically separate bright vehicles from the darker road.
    *   Find contours, filter out noise by area, and draw bounding boxes with a lane counter.
*   **Libraries:** `cv2`, `numpy`, `json`.

### Phase 4: Count vehicles
*  **Goal:** Count vehicles for each lane

### Phase 5: Video Processing Loop & Output
*   **Goal:** Apply the pipeline to the entire video and output the counting data.
*   **Techniques:** Loop through video frames, process them, draw bounding boxes and counters on the frame for visualization, and export the lane counts for the traffic light algorithm.
*   **Libraries:** `cv2`.
