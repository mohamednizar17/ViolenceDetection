import cv2
import numpy as np
import time
import mediapipe as mp # Import MediaPipe
import os # Import os for path validation
from google.colab.patches import cv2_imshow # Import cv2_imshow for Colab compatibility

# --- Configuration Parameters ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR_NORMAL = (0, 255, 0) # Green
TEXT_COLOR_ALERT = (0, 0, 255)  # Red
DISPLAY_FRAME_DELAY_MS = 1 # Delay for imshow (1ms for video playback), set to 1 for smooth video.

# --- MediaPipe Pose Setup ---
# NOTE: If you encounter "NameError: name 'audio_classifier' is not defined" related to MediaPipe,
# it's usually an issue with the MediaPipe installation itself.
# Please try uninstalling and reinstalling MediaPipe:
# pip uninstall mediapipe
# pip install mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Initialize MediaPipe Pose model with confidence thresholds
# min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for the pose detection to be considered successful.
# min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked successfully.
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Store previous frame's landmark positions for velocity calculation
prev_landmarks_positions = {}
# frame_count is not strictly needed for this logic.
last_alert_time = 0 # Used for temporary on-screen alert display

# --- Heuristics for "Aggressive Behavior" (These are critical and need careful tuning!) ---
# These thresholds define what constitutes "aggressive" movement based on pose analysis.
# They are relative and will likely need adjustment based on typical video content and expected fight intensity.

# VELOCITY_THRESHOLD: Normalized pixel velocity (0 to 1 range for x/y)
# This value determines how fast a joint needs to move to be considered "active".
# Lower value makes it more sensitive.
VELOCITY_THRESHOLD = 0.025 # Tuned: Slightly lower for more sensitivity

# ACTIVE_JOINTS_THRESHOLD: Number of joints moving above velocity threshold to trigger an alert.
# Lower value means fewer active joints are needed to detect aggression.
ACTIVE_JOINTS_THRESHOLD = 4 # Tuned: Lowered for more sensitivity


def calculate_joint_velocity(current_lm, prev_lm, width, height):
    """
    Calculates the velocity of a single joint between two frames.
    Velocity is normalized by the frame's diagonal length for scale invariance.

    Args:
        current_lm: The current landmark object (from MediaPipe).
        prev_lm: The previous landmark object.
        width: The width of the video frame.
        height: The height of the video frame.

    Returns:
        The normalized velocity as a float.
    """
    if not prev_lm:
        return 0.0

    # Denormalize coordinates to pixels for more accurate distance calculation
    current_x_px = current_lm.x * width
    current_y_px = current_lm.y * height
    prev_x_px = prev_lm.x * width
    prev_y_px = prev_lm.y * height

    dx = current_x_px - prev_x_px
    dy = current_y_px - prev_y_px
    # Euclidean distance
    velocity_px = np.sqrt(dx**2 + dy**2)
    # Normalize by the diagonal of the frame to make it scale-independent
    frame_diagonal = np.sqrt(width**2 + height**2)
    return velocity_px / frame_diagonal if frame_diagonal > 0 else 0


def run_fight_detection_from_video():
    """
    Prompts the user for a video file path, then performs human pose estimation using MediaPipe,
    analyzes joint movements for aggression, and provides a final verdict based on any single frame detection.
    """
    global prev_landmarks_positions, last_alert_time # Declare global to modify

    video_path = input("Please enter the path to the video file : ")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'. Please check the path and try again.")
        return

    print(f"Attempting to open video file: '{video_path}'...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'.")
        print("Please ensure the file is a valid video format and accessible.")
        return

    print("Video opened successfully. Analyzing video... (Press 'q' to quit during playback)")

    total_frames = 0
    overall_fight_detected = False # Flag to indicate if a fight was detected in ANY frame

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video stream or failed to grab frame
            break

        total_frames += 1

        frame_height, frame_width, _ = frame.shape

        # Convert the BGR image to RGB. MediaPipe processes RGB images.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image_rgb.flags.writeable = False

        # Process the image and get pose landmarks.
        results = pose.process(image_rgb)

        # Mark the image as writeable again after processing
        image_rgb.flags.writeable = True

        alert_message = "Analyzing..."
        alert_color = TEXT_COLOR_NORMAL # Initialize alert_color at the start of each loop iteration

        current_landmarks = None

        if results.pose_landmarks:
            current_landmarks = results.pose_landmarks.landmark
            # Draw the pose annotation (stick figure) on the original frame.
            # Draws both landmarks (points) and connections (lines)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Landmark color (orange)
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Connection color (pink/purple)
            )

            # --- Heuristics for "Aggressive Behavior" ---
            active_high_velocity_joints = 0
            # Define relevant joints for detecting aggressive motion (wrists, elbows, knees, ankles).
            relevant_joint_indices = [
                mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ]

            # Only calculate velocity if we have previous frame's landmarks
            if prev_landmarks_positions:
                for idx in relevant_joint_indices:
                    # Ensure both current and previous landmarks for this joint are visible (visibility > 0.7)
                    if (current_landmarks[idx].visibility > 0.7 and
                        idx in prev_landmarks_positions and
                        prev_landmarks_positions[idx].visibility > 0.7):

                        velocity = calculate_joint_velocity(
                            current_landmarks[idx],
                            prev_landmarks_positions[idx],
                            frame_width, frame_height
                        )
                        if velocity > VELOCITY_THRESHOLD:
                            active_high_velocity_joints += 1

            # Determine if movement is "aggressive" based on the count of highly active joints
            if active_high_velocity_joints >= ACTIVE_JOINTS_THRESHOLD:
                overall_fight_detected = True # Set the flag to True if aggression is found in ANY frame
                alert_message = f"!!! AGGRESSION DETECTED ({active_high_velocity_joints} active joints) !!!"
                alert_color = TEXT_COLOR_ALERT # Set alert_color here
            else:
                alert_message = "Analyzing..."
                alert_color = TEXT_COLOR_NORMAL # Set alert_color here
        else:
            # If no person is detected by MediaPipe Pose
            alert_message = "Analyzing (No person detected)"
            alert_color = TEXT_COLOR_NORMAL # Set alert_color here
            prev_landmarks_positions = {} # Clear previous landmarks if person disappears

        # Update previous landmarks for the next frame
        if current_landmarks:
            prev_landmarks_positions = {lm_id: current_landmarks[lm_id] for lm_id in range(len(current_landmarks))}

        # Display analysis status on the frame
        cv2.putText(frame, alert_message, (10, 30), FONT, 0.8, alert_color, 2, cv2.LINE_AA)

        # Display the processed frame using cv2_imshow for Colab compatibility
        cv2_imshow(frame)

        # Check for 'q' key press to exit the application
        if cv2.waitKey(DISPLAY_FRAME_DELAY_MS) & 0xFF == ord('q'):
            print("Analysis interrupted by user.")
            break

    # Release video file resources and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Video Analysis Complete ---")

    # --- Final Verdict Based on Overall Aggression Detection ---
    if total_frames == 0:
        print("No frames were processed from the video.")
        return

    if overall_fight_detected:
        print("\n*** VERDICT: FIGHT HAPPENED ***")
        print("Aggressive movement patterns were detected in at least one frame of the video.")
    else:
        print("\n*** VERDICT: NO FIGHT DETECTED ***")
        print("No aggressive movement patterns meeting the defined thresholds were detected in the video.")
    print("---------------------------------")


if __name__ == "__main__":
    run_fight_detection_from_video()
