import numpy as np


class RetargetingMapper:
    """
    Maps human pose landmarks to robot joint angles.
    Uses correct Simple Humanoid joint names.
    """
    
    # Joint order matches Humanoid.UPPER_BODY_JOINTS
    JOINT_ORDER = [
        "LARM_SHOULDER_P",   # Left shoulder pitch
        "LARM_SHOULDER_R",   # Left shoulder roll  
        "LARM_ELBOW",        # Left elbow
        "RARM_SHOULDER_P",   # Right shoulder pitch
        "RARM_SHOULDER_R",   # Right shoulder roll
        "RARM_ELBOW",        # Right elbow
    ]
    
    # Joint limits from URDF
    JOINT_LIMITS = [
        (-1.57, 0.79),   # LARM_SHOULDER_P
        (0.01, 2.87),    # LARM_SHOULDER_R
        (-2.23, 0.0),    # LARM_ELBOW (note: negative range)
        (-1.57, 0.79),   # RARM_SHOULDER_P
        (0.01, 2.87),    # RARM_SHOULDER_R
        (-2.23, 0.0),    # RARM_ELBOW
    ]
    
    def __init__(self, mirror=True):
        self.mirror = mirror
        self.calibration_offset = np.zeros(6)
        self.calibrated = False
        
    def calibrate(self, landmarks):
        """Calibrate with T-pose to set zero offsets."""
        if landmarks is None:
            return
        angles = self._compute_raw_angles(landmarks)
        self.calibration_offset = angles.copy()
        self.calibrated = True
        print(f"Calibrated with offsets: {self.calibration_offset}")
    
    def compute_angles(self, landmarks):
        """Convert pose landmarks to robot joint angles."""
        if landmarks is None:
            return None
        
        raw_angles = self._compute_raw_angles(landmarks)
        
        if self.calibrated:
            angles = raw_angles - self.calibration_offset
        else:
            angles = raw_angles
        
        return self._clamp_angles(angles)
    
    def _compute_raw_angles(self, landmarks):
        """Compute raw joint angles from landmark positions."""
        # Extract positions
        left_shoulder = np.array(landmarks.get("left_shoulder", [0, 0, 0]))
        left_elbow = np.array(landmarks.get("left_elbow", [0, 0, 0]))
        left_wrist = np.array(landmarks.get("left_wrist", [0, 0, 0]))
        right_shoulder = np.array(landmarks.get("right_shoulder", [0, 0, 0]))
        right_elbow = np.array(landmarks.get("right_elbow", [0, 0, 0]))
        right_wrist = np.array(landmarks.get("right_wrist", [0, 0, 0]))
        
        # Compute arm vectors
        left_upper = left_elbow - left_shoulder
        left_lower = left_wrist - left_elbow
        right_upper = right_elbow - right_shoulder
        right_lower = right_wrist - right_elbow
        
        # Compute joint angles
        # Shoulder pitch: forward/backward rotation
        left_shoulder_p = self._compute_shoulder_pitch(left_upper)
        right_shoulder_p = self._compute_shoulder_pitch(right_upper)
        
        # Shoulder roll: lateral rotation
        left_shoulder_r = self._compute_shoulder_roll(left_upper)
        right_shoulder_r = self._compute_shoulder_roll(right_upper)
        
        # Elbow: flexion angle
        left_elbow_angle = self._compute_elbow_angle(left_upper, left_lower)
        right_elbow_angle = self._compute_elbow_angle(right_upper, right_lower)
        
        if self.mirror:
            # When mirroring, swap left/right for robot
            return np.array([
                right_shoulder_p,      # Robot left = human right
                right_shoulder_r,
                right_elbow_angle,
                left_shoulder_p,       # Robot right = human left
                left_shoulder_r,
                left_elbow_angle
            ])
        else:
            return np.array([
                left_shoulder_p,
                left_shoulder_r,
                left_elbow_angle,
                right_shoulder_p,
                right_shoulder_r,
                right_elbow_angle
            ])
    
    def _compute_shoulder_pitch(self, upper_arm):
        """Compute shoulder pitch (forward/back) from upper arm vector."""
        norm = np.linalg.norm(upper_arm)
        if norm < 1e-6:
            return 0.0
        
        # MediaPipe: Y is down, Z is forward (away from camera)
        # Pitch is rotation around X axis (side-to-side axis)
        # Angle from vertical (negative Y direction)
        pitch = np.arctan2(upper_arm[2], -upper_arm[1])
        return float(pitch)
    
    def _compute_shoulder_roll(self, upper_arm):
        """Compute shoulder roll (lateral) from upper arm vector."""
        norm = np.linalg.norm(upper_arm)
        if norm < 1e-6:
            return 0.5  # Default slightly raised
        
        # Roll is how far the arm is from the body laterally
        # X component relative to YZ plane
        roll = np.arctan2(abs(upper_arm[0]), np.sqrt(upper_arm[1]**2 + upper_arm[2]**2))
        return float(roll) + 0.3  # Offset to keep arms slightly out
    
    def _compute_elbow_angle(self, upper, lower):
        """Compute elbow flexion angle."""
        norm_u = np.linalg.norm(upper)
        norm_l = np.linalg.norm(lower)
        if norm_u < 1e-6 or norm_l < 1e-6:
            return -0.5  # Slight bend
        
        upper_n = upper / norm_u
        lower_n = lower / norm_l
        
        # Angle between upper and lower arm
        dot = np.clip(np.dot(upper_n, lower_n), -1.0, 1.0)
        angle = np.arccos(dot)
        
        # Convert to elbow flexion (0 = straight, negative = bent)
        elbow = -(np.pi - angle)
        return float(elbow)
    
    def _clamp_angles(self, angles):
        """Clamp angles to joint limits."""
        clamped = np.zeros_like(angles)
        for i, (angle, (lo, hi)) in enumerate(zip(angles, self.JOINT_LIMITS)):
            clamped[i] = np.clip(angle, lo, hi)
        return clamped
