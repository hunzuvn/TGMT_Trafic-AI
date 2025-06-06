import cv2
import numpy as np

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)

    # Vùng quan tâm (ROI) - tam giác ở dưới cùng ảnh
    polygon = np.array([
        [(0, height), (width, height), (width, int(height*0.6)), (0, int(height*0.6))]
    ])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Phát hiện các đường thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return frame
