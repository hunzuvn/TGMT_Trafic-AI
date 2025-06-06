# modules/traffic_light.py
from ultralytics import YOLO
import cv2

# Load YOLO model (có thể thay thế bằng đường dẫn model riêng nếu cần)
model = YOLO("yolov5s.pt")  # Huấn luyện lại nếu cần class traffic light

# Màu sắc RGB cho từng trạng thái đèn
COLOR_MAP = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0)
}

def detect_traffic_light(frame):
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]

        if label == 'traffic light':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]  # vùng đèn
            light_color = get_light_color(roi)

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_MAP.get(light_color, (255,255,255)), 2)
            cv2.putText(frame, f"{light_color.upper()} LIGHT", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP.get(light_color, (255,255,255)), 2)
    return frame

def get_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))
    yellow_mask = cv2.inRange(hsv, (20, 70, 50), (35, 255, 255))

    red_val = red_mask.sum()
    green_val = green_mask.sum()
    yellow_val = yellow_mask.sum()

    max_color = max([(red_val, 'red'), (yellow_val, 'yellow'), (green_val, 'green')], key=lambda x: x[0])
    return max_color[1] if max_color[0] > 10000 else 'unknown'
