import cv2
import numpy as np

def order_points(pts):
    # Initializes a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def main():
    img_path = 'sample_img/sample.png'
    output_path = 'sample_img/phase1/phase1_output.png'
    
    print(f"Loading image from {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Try different dictionaries if needed, starting with DICT_4X4_50
    dict_to_try = [
        cv2.aruco.DICT_4X4_50, 
        cv2.aruco.DICT_4X4_100, 
        cv2.aruco.DICT_4X4_250, 
        cv2.aruco.DICT_4X4_1000
    ]
    
    corners = None
    ids = None
    
    for dict_id in dict_to_try:
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

        if ids is not None and len(ids) >= 4:
            print(f"Detected {len(ids)} markers using dictionary ID {dict_id}.")
            break

    if ids is None or len(ids) < 4:
        print("Error: Could not detect at least 4 markers.")
        if ids is not None:
            print(f"Detected IDs: {ids.flatten()}")
            
        # Draw whatever was detected for debugging
        debug_img = img.copy()
        if ids is not None:
            try:
                cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            except AttributeError:
                pass # older opencv version compatibility issue with drawDetectedMarkers
        cv2.imwrite('debug_markers.jpg', debug_img)
        print("Saved debug_markers.jpg to see what went wrong.")
        return

    # If we found 4 or more markers, take the first 4 for the perspective transform
    # Calculate the center of each marker
    centers = []
    for corner in corners[:4]:
        c = corner[0]
        center_x = c[:, 0].mean()
        center_y = c[:, 1].mean()
        centers.append([center_x, center_y])
    
    centers = np.array(centers, dtype="float32")
    
    # Order points to Top-Left, Top-Right, Bottom-Right, Bottom-Left
    src_pts = order_points(centers)
    (tl, tr, br, bl) = src_pts
    
    print("Ordered Source Points (TL, TR, BR, BL):")
    for pt in src_pts:
        print(f"  ({pt[0]:.1f}, {pt[1]:.1f})")

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the top-down view
    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    print(f"Calculating perspective transform matrix for {maxWidth}x{maxHeight} output...")
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective warp
    print("Applying warp perspective...")
    warped_img = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))

    # Save the output image
    cv2.imwrite(output_path, warped_img)
    print(f"Successfully saved {output_path}")

if __name__ == "__main__":
    main()
