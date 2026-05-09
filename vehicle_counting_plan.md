# Vehicle Counting Pipeline for Adaptive Traffic Lights

## Project Objective
Develop a robust computer vision pipeline to count the number of vehicles (represented by red and green blocks) in each lane from a video recording of a traffic simulation board (`sa_ban_traffic.mp4`). This data will subsequently be used to feed an adaptive traffic light control algorithm.

## Pipeline Evaluation
The proposed pipeline is highly optimal and perfectly suited for this specific scenario. Here is why:
1. **ArUco Markers:** The presence of ArUco markers at the four corners of the board is a massive advantage. It allows for an exact, mathematically precise perspective transformation. This eliminates errors caused by camera angle or minor camera movements.
2. **Adaptive Color Segmentation & Static ROIs:** Since Step 1 guarantees a fixed perspective, the physical coordinates of the lanes will never change. By defining Static ROIs early, we can crop out the lanes and eliminate 90% of background noise. Furthermore, processing colors adaptively per ROI (using color ratios) rather than relying on global HSV thresholds makes the pipeline incredibly robust against dynamic illumination shifts and color casts.

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

### Phase 3: Adaptive Color Segmentation per ROI
*   **Goal:** Isolate the red and green vehicles dynamically without using rigid global thresholds, adapting to local lighting shifts.
*   **Techniques:**
    *   Process each lane ROI independently.

### Phase 4: Vehicle Detection & Counting
*   **Goal:** Count the vehicles in their respective lanes using the generated masks.
*   **Techniques:**
    *   Find contours on the adaptively cleaned binary masks within each ROI.
    *   Calculate the centroid or bounding box of each vehicle contour.
    *   Filter out small noise contours by area.
    *   Increment the respective lane's counter.
*   **Libraries:** `cv2`.

### Phase 5: Video Processing Loop & Output
*   **Goal:** Apply the pipeline to the entire video and output the counting data.
*   **Techniques:** Loop through video frames, process them, draw bounding boxes and counters on the frame for visualization, and export the lane counts for the traffic light algorithm.
*   **Libraries:** `cv2`.
