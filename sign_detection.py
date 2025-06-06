# modules/sign_detection.py
from ultralytics import YOLO
import cv2

# Load mô hình đã huấn luyện nhận diện biển báo giao thông Việt Nam
model = YOLO("models/best.pt")  # Đặt file best.pt trong thư mục models/

def detect_sign(frame):
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id] if model.names else f"Class {cls_id}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return frame
