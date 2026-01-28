import numpy as np


class RetargetingMapper:
    
    def __init__(self, mirror=True):
        self.mirror = mirror
        self.calibration_offset = np.zeros(6)
        self.calibrated = False
        
    def calibrate(self, landmarks):
        if landmarks is None:
            return
        angles = self._compute_raw_angles(landmarks)
        self.calibration_offset = angles
        self.calibrated = True
    
    def compute_angles(self, landmarks):
        if landmarks is None:
            return None
        raw_angles = self._compute_raw_angles(landmarks)
        if self.calibrated:
            angles = raw_angles - self.calibration_offset
        else:
            angles = raw_angles
        return self._clamp_angles(angles)
    
    def _compute_raw_angles(self, landmarks):
        left_shoulder = landmarks.get("left_shoulder", np.zeros(3))
        left_elbow = landmarks.get("left_elbow", np.zeros(3))
        left_wrist = landmarks.get("left_wrist", np.zeros(3))
        right_shoulder = landmarks.get("right_shoulder", np.zeros(3))
        right_elbow = landmarks.get("right_elbow", np.zeros(3))
        right_wrist = landmarks.get("right_wrist", np.zeros(3))
        
        left_upper = left_elbow - left_shoulder
        left_lower = left_wrist - left_elbow
        right_upper = right_elbow - right_shoulder
        right_lower = right_wrist - right_elbow
        
        left_shoulder_pitch = self._compute_pitch(left_upper)
        left_shoulder_roll = self._compute_roll(left_upper)
        left_elbow_angle = self._compute_elbow_angle(left_upper, left_lower)
        
        right_shoulder_pitch = self._compute_pitch(right_upper)
        right_shoulder_roll = self._compute_roll(right_upper)
        right_elbow_angle = self._compute_elbow_angle(right_upper, right_lower)
        
        if self.mirror:
            return np.array([
                -right_shoulder_pitch,
                -right_shoulder_roll,
                -right_elbow_angle,
                -left_shoulder_pitch,
                left_shoulder_roll,
                left_elbow_angle
            ])
        else:
            return np.array([
                left_shoulder_pitch,
                left_shoulder_roll,
                left_elbow_angle,
                right_shoulder_pitch,
                right_shoulder_roll,
                right_elbow_angle
            ])
    
    def _compute_pitch(self, vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return 0.0
        vec_n = vec / norm
        pitch = np.arctan2(-vec_n[1], -vec_n[2])
        return pitch
    
    def _compute_roll(self, vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return 0.0
        vec_n = vec / norm
        roll = np.arctan2(vec_n[0], np.sqrt(vec_n[1]**2 + vec_n[2]**2))
        return roll
    
    def _compute_elbow_angle(self, upper, lower):
        norm_u = np.linalg.norm(upper)
        norm_l = np.linalg.norm(lower)
        if norm_u < 1e-6 or norm_l < 1e-6:
            return 0.0
        upper_n = upper / norm_u
        lower_n = lower / norm_l
        dot = np.clip(np.dot(upper_n, lower_n), -1.0, 1.0)
        angle = np.arccos(dot)
        elbow = np.pi - angle
        return elbow
    
    def _clamp_angles(self, angles):
        limits = [
            (-3.14, 0.5),
            (-0.5, 2.5),
            (-2.5, 0.0),
            (-3.14, 0.5),
            (-2.5, 0.5),
            (0.0, 2.5)
        ]
        clamped = np.zeros_like(angles)
        for i, (a, (lo, hi)) in enumerate(zip(angles, limits)):
            clamped[i] = np.clip(a, lo, hi)
        return clamped
