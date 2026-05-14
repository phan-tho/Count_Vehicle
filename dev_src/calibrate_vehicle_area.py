import cv2
import numpy as np
import json
import os

MASK_IMAGE_PATH = 'sample_img/phase3/phase3_grayscale_mask.png'
PARAMS_JSON_PATH = 'vehicle_params.json'

def main():
    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"Error: {MASK_IMAGE_PATH} not found.")
        return

    img = cv2.imread(MASK_IMAGE_PATH)
    
    print("======================================================")
    print("HƯỚNG DẪN HIỆU CHỈNH KÍCH THƯỚC XE:")
    print("1. Kéo thả chuột để vẽ hình chữ nhật bao quanh 1 chiếc xe (ô tô hoặc xe máy).")
    print("2. Bấm ENTER hoặc SPACE để xác nhận hình chữ nhật đó.")
    print("3. Lặp lại bước 1 và 2 cho khoảng 5-10 phương tiện (cả ô tô và xe máy).")
    print("4. Bấm ESC khi bạn đã chọn xong tất cả.")
    print("======================================================")

    window_name = "Calibrate Vehicle Area (Press ESC to finish)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)

    # selectROIs returns a tuple of bounding boxes: (x, y, w, h)
    rois = cv2.selectROIs(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if len(rois) == 0:
        print("Bạn chưa chọn phương tiện nào. Thoát chương trình.")
        return

    areas = []
    print("\n--- KẾT QUẢ ĐO ĐẠC ---")
    for i, (x, y, w, h) in enumerate(rois):
        area = w * h
        areas.append(area)
        print(f"Phương tiện {i+1}: Kích thước {w}x{h} -> Diện tích = {area} pixels")

    min_area = min(areas)
    max_area = max(areas)
    avg_area = sum(areas) / len(areas)

    # Đề xuất các thông số
    # Lấy diện tích nhỏ nhất chia đôi làm ngưỡng để lọc nhiễu (bụi, nhiễu trắng)
    suggested_min_threshold = int(min_area * 0.5) 
    
    # Lấy diện tích trung bình làm chuẩn để đếm xe
    suggested_average_area = int(avg_area)

    print("\n--- THÔNG SỐ ĐỀ XUẤT CHO PHASE 4 ---")
    print(f"MIN_VEHICLE_AREA (Ngưỡng lọc nhiễu): {suggested_min_threshold}")
    print(f"AVERAGE_VEHICLE_AREA (Diện tích trung bình 1 xe): {suggested_average_area}")

    # Lưu lại vào file JSON để Phase 4 có thể tự động đọc
    params = {
        "MIN_VEHICLE_AREA": suggested_min_threshold,
        "AVERAGE_VEHICLE_AREA": suggested_average_area,
        "SAMPLES_COUNT": len(areas),
        "RAW_AREAS": areas
    }

    with open(PARAMS_JSON_PATH, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"\nĐã lưu các thông số tự động vào file '{PARAMS_JSON_PATH}'!")

if __name__ == "__main__":
    main()
