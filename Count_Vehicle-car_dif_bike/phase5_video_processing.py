import cv2
import numpy as np
import json
import os
import csv

def order_points(pts):
    # Initializes a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def get_perspective_matrix(img):
    corners = None
    ids = None
    
    dict_id = cv2.aruco.DICT_4X4_50
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(dict_id)

    try:
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(img)
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) < 4:
        return None, None

    centers = []
    for corner in corners[:4]:
        c = corner[0]
        center_x = c[:, 0].mean()
        center_y = c[:, 1].mean()
        centers.append([center_x, center_y])
    
    centers = np.array(centers, dtype="float32")
    src_pts = order_points(centers)
    (tl, tr, br, bl) = src_pts
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return matrix, (maxWidth, maxHeight)

def process_roi_grayscale(gray):
    if np.std(gray) < 10 or np.max(gray) < 120:
        return np.zeros_like(gray, dtype=np.uint8)

    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if ret < np.median(gray) + 15:
        return np.zeros_like(gray, dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def main():
    # Resolve all paths relative to this script's directory
    # so the script works correctly regardless of CWD
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(BASE_DIR, 'sa_ban_traffic.mp4')
    OUT_VIDEO_PATH = os.path.join(BASE_DIR, 'phase5_output_video.mp4')
    OUT_CSV_PATH = os.path.join(BASE_DIR, 'phase5_traffic_data.csv')

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
        return
    rois = [[6,282,158,131],[436,290,1099,123],[1810,167,153,119],
    [423,164,1112,122],[6,746,150,123],[426,750,1113,118],[1810,624,156,121],
    [430,622,1107,117],[288,877,134,154],[296,419,128,202],[165,3,133,155],
    [169,422,123,194],[1668,884,138,142],[1676,423,124,189],[1547,6,130,156],
    [1549,413,118,197]]

    MIN_VEHICLE_AREA = 162
    AVERAGE_CAR_AREA = 895
    AVERAGE_BIKE_AREA = 400

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Default if can't read
    
    matrix = None
    out_size = None
    
    # Read frames until we find a perspective matrix
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read video or no ArUco markers found in any frame.")
            return
            
        matrix, out_size = get_perspective_matrix(frame)
        if matrix is not None:
            break
            
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, out_size)
    
    csv_file = open(OUT_CSV_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Prepare CSV Header
    header = ['Frame']
    for i in range(len(rois)):
        header.extend([f'Lane_{i}_Cars', f'Lane_{i}_Bikes'])
    csv_writer.writerow(header)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        warped_img = cv2.warpPerspective(frame, matrix, out_size)
        output_img = warped_img.copy()
        
        frame_data = [frame_count]
        
        for i, roi in enumerate(rois):
            x, y, w, h = roi
            # Clamp ROI to the warped image bounds
            img_h, img_w = warped_img.shape[:2]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, img_w), min(y + h, img_h)
            roi_bgr = warped_img[y1:y2, x1:x2]
            # Skip if ROI is empty (outside frame bounds)
            if roi_bgr.size == 0:
                frame_data.extend([0, 0])
                continue
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            roi_mask = process_roi_grayscale(gray)
            
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lane_cars = 0
            lane_bikes = 0
            
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_VEHICLE_AREA:
                    continue
                    
                single_mask = np.zeros(roi_mask.shape, dtype=np.uint8)
                cv2.drawContours(single_mask, [cnt], -1, 255, -1)
                
                b = roi_bgr[:, :, 0].astype(np.int16)
                g = roi_bgr[:, :, 1].astype(np.int16)
                r = roi_bgr[:, :, 2].astype(np.int16)
                
                in_mask = single_mask > 0
                
                red_cond = (r > g + 20) & (r > b + 20) & in_mask
                red_px = np.sum(red_cond)
                
                green_cond = (g > r + 20) & (g > b + 20) & in_mask
                green_px = np.sum(green_cond)
                
                total_color_px = red_px + green_px
                if total_color_px == 0:
                    cars = 0
                    bikes = 1
                else:
                    r_ratio = red_px / total_color_px
                    g_ratio = green_px / total_color_px
                    
                    cars = max(0, round((area * r_ratio) / AVERAGE_CAR_AREA))
                    bikes = max(0, round((area * g_ratio) / AVERAGE_BIKE_AREA))
                
                lane_cars += cars
                lane_bikes += bikes
                
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                global_x = x + cx
                global_y = y + cy
                
                cv2.rectangle(output_img, (global_x, global_y), (global_x + cw, global_y + ch), (0, 255, 255), 2)
                
                label = f"C:{cars} B:{bikes}"
                text_y = global_y - 5 if global_y - 5 > 10 else global_y + 15
                cv2.putText(output_img, label, (global_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                
            lane_label = f"Lane {i}: Cars={lane_cars} Bikes={lane_bikes}"
            lane_text_y = y - 10 if y - 10 > 20 else y + h + 20
            cv2.putText(output_img, lane_label, (x, lane_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            frame_data.extend([lane_cars, lane_bikes])
            
        csv_writer.writerow(frame_data)
        out_video.write(output_img)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out_video.release()
    csv_file.close()
    
    print(f"Successfully processed {frame_count} frames.")
    print(f"Output video saved to {OUT_VIDEO_PATH}")
    print(f"Traffic data saved to {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()
