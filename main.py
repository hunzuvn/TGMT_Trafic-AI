import cv2
from modules.traffic_light import detect_traffic_light
from modules.sign_detection import detect_sign
from modules.pose_estimation import detect_pose
from modules.lane_detection import detect_lanes

cap = cv2.VideoCapture(0)  # Mở webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_traffic_light(frame)
    frame = detect_sign(frame)
    frame = detect_pose(frame)
    frame = detect_lanes(frame)

    cv2.imshow("Traffic AI App", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
