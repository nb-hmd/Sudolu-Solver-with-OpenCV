import cv2
import numpy as np

def pre_process_image(img):
    """
    Grayscale, Blur, and Threshold the image
    """
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    # Adaptive threshold is often better for varying lighting conditions
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    # Dilate to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh


def pre_process_grid_for_ocr(img):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        5,
    )

    cleaned = remove_grid_lines(thresh)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    cleaned = cv2.medianBlur(cleaned, 3)
    return cleaned

def find_contours(img, original):
    """
    Find contours in the thresholded image and return the biggest one (the grid)
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 2000: # Filter out small noise
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(myPoints):
    """
    Reorder points to: top-left, top-right, bottom-left, bottom-right
    Wait, let's stick to the order used in get_warp: TL, TR, BL, BR
    """
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 2), dtype=np.int32)
    add = myPoints.sum(1)
    
    myPointsNew[0] = myPoints[np.argmin(add)]  # TL
    myPointsNew[3] = myPoints[np.argmax(add)]  # BR
    
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # TR
    myPointsNew[2] = myPoints[np.argmax(diff)] # BL
    
    return myPointsNew

def get_warp(img, biggest, width, height):
    """
    Apply perspective transform to get a top-down view of the grid
    Expects biggest to be [TL, TR, BL, BR]
    """
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    return imgOutput

def split_boxes(img):
    """
    Split the grid image into 81 small images (cells)
    """
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

def remove_grid_lines(img):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    cell = max(1, gray.shape[1] // 9)
    k = max(25, cell * 3)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))

    h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h, iterations=1)
    v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v, iterations=1)
    grid = cv2.bitwise_or(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=2)

    cleaned = cv2.bitwise_and(gray, cv2.bitwise_not(grid))
    return cleaned

def display_numbers(img, numbers, color=(0, 255, 0)):
    """
    Overlay the solution numbers on the image.
    This is a simplified visualization.
    """
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                 cv2.putText(img, str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return img


def display_numbers_with_confidence(img, numbers, confidences, digit_color=(0, 255, 0), conf_color=(0, 165, 255)):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(0, 9):
        for y in range(0, 9):
            idx = (y * 9) + x
            val = numbers[idx]
            if val != 0:
                cv2.putText(
                    img,
                    str(val),
                    (x * secW + int(secW / 2) - 10, int((y + 0.72) * secH)),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    digit_color,
                    2,
                    cv2.LINE_AA,
                )

            conf = confidences[idx]
            if conf > 0:
                cv2.putText(
                    img,
                    f"{conf:.2f}",
                    (x * secW + 3, y * secH + 14),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    conf_color,
                    1,
                    cv2.LINE_AA,
                )
    return img
