import mediapipe as mp
import numpy as np


class PoseEstimator:
    
    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    BOXING_LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24
    }
    
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def estimate(self, rgb_frame):
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks is None:
            return None
        landmarks = np.zeros((33, 4))
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i, 0] = lm.x
            landmarks[i, 1] = lm.y
            landmarks[i, 2] = lm.z
            landmarks[i, 3] = lm.visibility
        return landmarks
    
    def get_boxing_landmarks(self, landmarks):
        if landmarks is None:
            return None
        result = {}
        for name, idx in self.BOXING_LANDMARKS.items():
            result[name] = landmarks[idx, :3]
        return result
    
    def get_world_landmarks(self, rgb_frame):
        results = self.pose.process(rgb_frame)
        if results.pose_world_landmarks is None:
            return None
        landmarks = np.zeros((33, 4))
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            landmarks[i, 0] = lm.x
            landmarks[i, 1] = lm.y
            landmarks[i, 2] = lm.z
            landmarks[i, 3] = lm.visibility
        return landmarks
    
    def close(self):
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
