# modules/pose_estimation.py
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Xác định tín hiệu dựa trên tư thế tay cơ bản
# (Có thể mở rộng hoặc tùy chỉnh theo quy định giao thông Việt Nam)
def interpret_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    h_gesture = "Không rõ tín hiệu"

    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        h_gesture = "Giơ cả hai tay lên – Dừng lại"
    elif left_wrist.y < left_shoulder.y:
        h_gesture = "Giơ tay trái – Rẽ trái"
    elif right_wrist.y < right_shoulder.y:
        h_gesture = "Giơ tay phải – Rẽ phải"

    return h_gesture

def detect_pose(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        action = interpret_pose(landmarks)

        cv2.putText(frame, action, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame
