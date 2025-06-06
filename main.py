import cv2
from modules.traffic_light import detect_traffic_light
from modules.sign_detection import detect_sign
from modules.pose_estimation import detect_pose
from modules.lane_detection import detect_lanes
import time

from modules.camera_module import Camera
cam = Camera()
frame = cam.get_frame()

def main():
    cap = cv2.VideoCapture(0)  # Mở webcam

    if not cap.isOpened():
        print("Không mở được camera.")
        return

    # Khởi tạo biến đo FPS
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được khung hình từ camera.")
            break

        # Gọi các module nhận diện
        frame = detect_traffic_light(frame)
        frame = detect_sign(frame)
        frame = detect_pose(frame)
        frame = detect_lanes(frame)

        # Hiển thị FPS
        frame_count += 1
        if frame_count >= 10:
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0
        else:
            fps = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị kết quả
        cv2.imshow("Traffic AI App", frame)

        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
