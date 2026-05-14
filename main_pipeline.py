import cv2
import numpy as np
import json
import os
import time
import csv

# --- HYPERPARAMETERS ---
VIDEO_PATH = 'sa_ban_traffic.mp4'  # Use sample.png for image testing, sa_ban_traffic.mp4 for video
OUTPUT_VIDEO_PATH = 'output_traffic.mp4'
OUTPUT_CSV_PATH = 'traffic_data.csv'
PROCESSED_FPS = 5  # Target FPS for processing (to simulate Pi5 performance)
PERSPECTIVE_INTERVAL = 5  # Recompute perspective every N processed frames (helps with camera shake)
ROIS_PATH = 'lane_rois.json'
PARAMS_PATH = 'vehicle_params.json'
# -----------------------

class VehicleCounter:
    def __init__(self, rois_path=ROIS_PATH, params_path=PARAMS_PATH):
        self.matrix = None
        self.max_width = 0
        self.max_height = 0
        
        if os.path.exists(rois_path):
            with open(rois_path, 'r') as f:
                self.rois = json.load(f)
        else:
            print(f"Warning: {rois_path} not found.")
            self.rois = []
            
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                self.min_vehicle_area = params.get("MIN_VEHICLE_AREA", 200)
                self.car_area = params.get("AVERAGE_CAR_AREA", 895)
                self.bike_area = params.get("AVERAGE_BIKE_AREA", 400)
        else:
            print(f"Warning: {params_path} not found. Using defaults.")
            self.min_vehicle_area = 200
            self.car_area = 895
            self.bike_area = 400

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
        
    """
    [
    {'lane': 0, 'cars': 0, 'bikes': 0}, 
    {'lane': 1, 'cars': 0, 'bikes': 0}, 
    {'lane': 3, 'cars': 0, 'bikes': 2}, 
    {'lane': 10, 'cars': 0, 'bikes': 2}]
    """

    def init_perspective(self, img):
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
            return False
            
        centers = []
        for corner in corners[:4]:
            c = corner[0]
            center_x = c[:, 0].mean()
            center_y = c[:, 1].mean()
            centers.append([center_x, center_y])
        
        centers = np.array(centers, dtype="float32")
        src_pts = self.order_points(centers)
        (tl, tr, br, bl) = src_pts
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        self.max_width = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        self.max_height = max(int(heightA), int(heightB))

        dst_pts = np.array([
            [0, 0],
            [self.max_width - 1, 0],
            [self.max_width - 1, self.max_height - 1],
            [0, self.max_height - 1]
        ], dtype="float32")
        
        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return True

    def process_roi_grayscale(self, gray):
        if np.std(gray) < 10 or np.max(gray) < 120:
            return np.zeros_like(gray, dtype=np.uint8)

        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if ret < np.median(gray) + 15:
            return np.zeros_like(gray, dtype=np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
        
    def process_frame(self, img, recompute_perspective=False):
        # 1. Perspective Transform (Phase 1)
        if self.matrix is None or recompute_perspective:
            success = self.init_perspective(img)
            if not success and self.matrix is None:
                # Cannot initialize perspective and no previous matrix, return original image
                return img, []
                
        warped_img = cv2.warpPerspective(img, self.matrix, (self.max_width, self.max_height))
        output_img = warped_img.copy()
        
        results = []
        
        # 2 & 3 & 4. Process each ROI: mask, grayscale, counting
        for i, roi in enumerate(self.rois):
            x, y, w, h = roi
            roi_bgr = warped_img[y:y+h, x:x+w]
            
            if roi_bgr.shape[0] == 0 or roi_bgr.shape[1] == 0:
                continue
                
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            roi_mask = self.process_roi_grayscale(gray)
            
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lane_cars = 0
            lane_bikes = 0
            
            # Draw lane boundaries
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_vehicle_area:
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
                    
                    cars = max(0, round((area * r_ratio) / self.car_area))
                    bikes = max(0, round((area * g_ratio) / self.bike_area))
                
                lane_cars += cars
                lane_bikes += bikes
                
                # Draw vehicle bounding boxes
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
            
            results.append({'lane': i, 'cars': lane_cars, 'bikes': lane_bikes})
            
        return output_img, results

def test_single_image(img_path='sample_img/sample.png'):
    print(f"--- Processing Single Image: {img_path} ---")
    counter = VehicleCounter()
    img = cv2.imread("sample_img/debug_video/frame_0.jpg")
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
        
    out_img, results = counter.process_frame(img)
    print("Results for each lane:")
    for res in results:
        print(f"  Lane {res['lane']}: {res['cars']} Cars, {res['bikes']} Bikes")
        
    out_path = 'sample_img/debug_video/frame_0_outout.jpg'
    cv2.imwrite(out_path, out_img)
    print(f"Saved processed image to {out_path}\n")

def process_video():
    print(f"--- Processing Video: {VIDEO_PATH} ---")
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file {VIDEO_PATH} not found.")
        return
        
    counter = VehicleCounter()
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error opening video stream or file: {VIDEO_PATH}")
        return
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30 # fallback
        
    frame_interval = max(1, int(video_fps / PROCESSED_FPS))
    print(f"Source FPS: {video_fps:.1f}, Target FPS: {PROCESSED_FPS}, Processing every {frame_interval}th frame")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_idx = 0
    processed_count = 0
    
    # Open CSV file for writing
    with open(OUTPUT_CSV_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['Frame', 'Timestamp', 'Lane', 'Cars', 'Bikes'])
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                start_time = time.time()
                recompute = (processed_count % PERSPECTIVE_INTERVAL == 0)
                out_img, results = counter.process_frame(frame, recompute_perspective=recompute)
                end_time = time.time()

                if (frame_idx % 100 == 0):
                    cv2.imwrite(f"sample_img/debug_video/frame_{frame_idx}.jpg", frame)
                    cv2.imwrite(f"sample_img/debug_video/frame_{frame_idx}_out.jpg", out_img)
                
                process_ms = (end_time - start_time) * 1000
                
                # Here is where we would send data to the server, e.g.:
                # requests.post(SERVER_URL, json={"timestamp": time.time(), "counts": results})
                
                # print(f"Frame {frame_idx:04d} | Processed in {process_ms:.1f}ms | {results}")
                
                # Write to CSV
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                for res in results:
                    csv_writer.writerow([frame_idx, current_time, res['lane'], res['cars'], res['bikes']])
                
                if out is None and out_img is not None:
                    h, w = out_img.shape[:2]
                    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, PROCESSED_FPS, (w, h))
                    
                if out is not None and out_img is not None:
                    out.write(out_img)
                    
                processed_count += 1
                    
            frame_idx += 1
            
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Processed {processed_count} frames. Saved to {OUTPUT_VIDEO_PATH} and {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    test_single_image()
    # process_video()