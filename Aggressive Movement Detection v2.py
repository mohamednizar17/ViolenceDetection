import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Setup Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Parameters
MOTION_THRESHOLD = 50  # Threshold for aggressive motion
CONSECUTIVE_FRAMES_REQUIRED = 15  # Number of consecutive aggressive frames required

prev_landmarks = None
consecutive_aggressive_frames = 0

# Setup webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Cannot access webcam!"

def compute_motion(curr, prev):
    """Sum Euclidean distances for wrists and ankles."""
    points = [mp_pose.PoseLandmark.LEFT_WRIST,
              mp_pose.PoseLandmark.RIGHT_WRIST,
              mp_pose.PoseLandmark.LEFT_ANKLE,
              mp_pose.PoseLandmark.RIGHT_ANKLE]
    total = 0.0
    for p in points:
        idx = p.value
        dx = curr[idx].x - prev[idx].x
        dy = curr[idx].y - prev[idx].y
        total += np.hypot(dx, dy)
    return total * 1000  # scaling for sensitivity

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    fight_detected = False

    if result.pose_landmarks:
        curr = result.pose_landmarks.landmark
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if prev_landmarks:
            motion = compute_motion(curr, prev_landmarks)
            if motion > MOTION_THRESHOLD:
                consecutive_aggressive_frames += 1
            else:
                consecutive_aggressive_frames = 0

            fight_detected = consecutive_aggressive_frames >= CONSECUTIVE_FRAMES_REQUIRED

        prev_landmarks = curr
    else:
        consecutive_aggressive_frames = 0  # reset if no person detected

    status = "⚠️ FIGHT DETECTED!" if fight_detected else "No fight detected"
    color = (0, 0, 255) if fight_detected else (0, 255, 0)
    cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Consecutive Frame-Based Fight Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
