# Vehicle Counting Pipeline for Adaptive Traffic Lights

## Project Objective
Develop a robust computer vision pipeline to count the number of vehicles (represented by red and green blocks) in each lane from a video recording of a traffic simulation board (`sa_ban_traffic.mp4`). This data will subsequently be used to feed an adaptive traffic light control algorithm.

## Pipeline Evaluation
The proposed pipeline is highly optimal and perfectly suited for this specific scenario. Here is why:
1. **ArUco Markers:** The presence of ArUco markers at the four corners of the board is a massive advantage. It allows for an exact, mathematically precise perspective transformation. This eliminates errors caused by camera angle or minor camera movements.
2. **HSV Thresholding:** Given the distinct colors (red and green) of the "vehicles", HSV color space is the most resilient method against lighting variations, especially screen glare or moiré patterns present in the video.
3. **Static ROIs:** Since Step 1 guarantees a fixed, flattened perspective, the physical coordinates of the lanes will never change. Using static Region of Interests (ROIs) is computationally extremely cheap and highly reliable compared to complex object tracking algorithms (like SORT or DeepSORT).

---

## Phased Implementation Plan

### Phase 1: Perspective Transform & Alignment
*   **Goal:** Convert the angled camera view into a perfect, flat, top-down 2D view. Fix the coordinate system.
*   **Techniques:** 
    *   ArUco marker detection to find the 4 corners.
    *   Calculate the perspective transformation matrix.
    *   Warp the perspective to crop and flatten the board.
*   **Libraries:** `cv2` (OpenCV), `cv2.aruco`, `numpy`.

### Phase 2: Pre-processing (Noise Reduction)
*   **Goal:** Clean up the video feed, specifically reducing the screen's moiré effect and digital noise without altering the colors.
*   **Techniques:** Apply a subtle blur (e.g., Gaussian Blur) to smooth out the pixels. Avoid histogram equalization to prevent color distortion.
*   **Libraries:** `cv2`.

### Phase 3: Color Segmentation (HSV Thresholding)
*   **Goal:** Isolate the red and green vehicles from the dark background.
*   **Techniques:**
    *   Convert the flattened image from BGR to HSV color space.
    *   Define HSV bounds for 'Red' and 'Green'.
    *   Create binary masks for both colors.
    *   Apply morphological operations (Opening/Closing) to remove tiny noise specks and bridge small gaps in the vehicle blocks.
*   **Libraries:** `cv2`, `numpy`.

### Phase 4: Vehicle Detection & Static ROIs Mapping
*   **Goal:** Count the vehicles in their respective lanes.
*   **Techniques:**
    *   Find contours on the cleaned binary masks to detect individual vehicles.
    *   Calculate the centroid (center point) of each vehicle contour.
    *   Define static rectangular bounding boxes (ROIs) for each lane on the flattened image.
    *   Check which ROI contains the centroid of each vehicle and increment the respective lane's counter.
*   **Libraries:** `cv2`.

### Phase 5: Video Processing Loop & Output
*   **Goal:** Apply the pipeline to the entire video and output the counting data.
*   **Techniques:** Loop through video frames, process them, draw bounding boxes and counters on the frame for visualization, and export the lane counts for the traffic light algorithm.
*   **Libraries:** `cv2`.
