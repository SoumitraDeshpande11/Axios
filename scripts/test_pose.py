#!/usr/bin/env python3
"""
Test pose estimation without the robot.
Run this to verify your camera and MediaPipe are working.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import mediapipe as mp


def main():
    print("=" * 50)
    print("AXIOS - Pose Estimation Test")
    print("=" * 50)
    print("\nOpening camera...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    print("Camera opened successfully.")
    print("Press 'q' to quit.\n")
    
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Show key joints
                landmarks = results.pose_landmarks.landmark
                h, w = frame.shape[:2]
                
                # Left wrist
                lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                cv2.putText(frame, "L", (int(lw.x * w), int(lw.y * h)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Right wrist
                rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                cv2.putText(frame, "R", (int(rw.x * w), int(rw.y * h)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("AXIOS Pose Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("Done.")


if __name__ == "__main__":
    main()
