#!/usr/bin/env python3
"""
ATOM Mirror System - Real-time humanoid motion mirroring.
Robot copies your boxing movements via webcam.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import yaml
import cv2

from src.simulation.environment import AxiosEnv
from src.simulation.controller import TrajectoryInterpolator
from src.perception.camera import Camera
from src.perception.pose_estimator import PoseEstimator
from src.perception.filters import PoseFilter
from src.retargeting.mapper import RetargetingMapper


class AtomMirror:
    
    def __init__(self, config_path=None, camera_config_path=None, show_camera=True):
        self.config_path = config_path
        self.camera_config = self._load_camera_config(camera_config_path)
        self.show_camera = show_camera
        
        self.env = None
        self.camera = None
        self.pose_estimator = None
        self.pose_filter = None
        self.mapper = None
        self.interpolator = None
        self.running = False
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        
    def _load_camera_config(self, path):
        if path is None:
            return {}
        p = Path(path)
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def setup(self):
        print("Setting up ATOM Mirror System...")
        
        # PyBullet simulation
        print("  - Initializing PyBullet simulation")
        self.env = AxiosEnv(config_path=self.config_path, render=True)
        self.env.connect()
        
        # Camera
        print("  - Opening camera")
        cam_cfg = self.camera_config
        self.camera = Camera(
            device_index=cam_cfg.get("device_index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30)
        )
        if not self.camera.open():
            print("ERROR: Could not open camera!")
            return False
        
        # Pose estimation
        print("  - Loading MediaPipe pose model")
        pose_cfg = cam_cfg.get("pose_estimation", {})
        self.pose_estimator = PoseEstimator(
            model_complexity=pose_cfg.get("model_complexity", 1),
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5)
        )
        
        # Pose filter
        filter_cfg = cam_cfg.get("filter", {})
        self.pose_filter = PoseFilter(
            num_landmarks=33,
            num_coords=3,
            min_cutoff=filter_cfg.get("min_cutoff", 1.0),
            beta=filter_cfg.get("beta", 0.007)
        )
        
        # Retargeting
        self.mapper = RetargetingMapper(mirror=True)
        self.interpolator = TrajectoryInterpolator(num_joints=6, max_velocity=5.0)
        
        print("Setup complete!")
        return True
        
    def run(self):
        self.running = True
        print("\n" + "=" * 50)
        print("ATOM MIRROR RUNNING")
        print("=" * 50)
        print("1. Stand in T-pose for calibration (~1 sec)")
        print("2. Once calibrated, start boxing!")
        print("3. Press 'q' or Ctrl+C to quit")
        print("=" * 50 + "\n")
        
        calibrated = False
        calibration_frames = 0
        calibration_target = 30
        frame_count = 0
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get camera frame
                frame, timestamp = self.camera.get_frame_rgb()
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Run pose estimation
                landmarks = self.pose_estimator.estimate(frame)
                
                # Display camera if enabled
                if self.show_camera:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    display_frame = cv2.flip(display_frame, 1)  # Mirror
                    
                    if landmarks is not None:
                        self._draw_pose(display_frame, landmarks)
                        status = "CALIBRATING..." if not calibrated else "TRACKING"
                        color = (0, 255, 255) if not calibrated else (0, 255, 0)
                    else:
                        status = "NO POSE DETECTED"
                        color = (0, 0, 255)
                    
                    cv2.putText(display_frame, status, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.imshow("ATOM Mirror - Camera", display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if landmarks is None:
                    self.env.step(8)
                    continue
                
                # Filter pose
                filtered = self.pose_filter.filter(landmarks, timestamp)
                boxing_landmarks = self._extract_boxing_landmarks(filtered)
                
                # Calibration phase
                if not calibrated:
                    calibration_frames += 1
                    if calibration_frames >= calibration_target:
                        self.mapper.calibrate(boxing_landmarks)
                        self.interpolator.reset(np.zeros(6))
                        calibrated = True
                        print("\n>>> CALIBRATED! Start moving! <<<\n")
                    self.env.step(8)
                    continue
                
                # Compute joint angles and apply to robot
                joint_angles = self.mapper.compute_angles(boxing_landmarks)
                if joint_angles is not None:
                    self.interpolator.set_target(joint_angles)
                    smoothed = self.interpolator.update(self.frame_time)
                    self.env.set_joint_targets(smoothed)
                
                self.env.step(8)
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.frame_time - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def _extract_boxing_landmarks(self, landmarks):
        """Extract boxing-relevant landmarks from full pose."""
        indices = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24
        }
        result = {}
        for name, idx in indices.items():
            result[name] = landmarks[idx]
        return result
    
    def _draw_pose(self, frame, landmarks):
        """Draw pose landmarks on frame."""
        h, w = frame.shape[:2]
        
        # Key points to draw
        points = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
        
        # Draw connections
        for start, end in connections:
            x1 = int((1 - landmarks[start, 0]) * w)  # Flip X
            y1 = int(landmarks[start, 1] * h)
            x2 = int((1 - landmarks[end, 0]) * w)
            y2 = int(landmarks[end, 1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw points
        for idx in points:
            x = int((1 - landmarks[idx, 0]) * w)
            y = int(landmarks[idx, 1] * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    def stop(self):
        self.running = False
        if self.show_camera:
            cv2.destroyAllWindows()
        if self.camera is not None:
            self.camera.release()
        if self.pose_estimator is not None:
            self.pose_estimator.close()
        if self.env is not None:
            self.env.close()
        print("ATOM Mirror stopped.")


def main():
    config_path = PROJECT_ROOT / "config" / "humanoid.yaml"
    camera_config_path = PROJECT_ROOT / "config" / "camera.yaml"
    
    mirror = AtomMirror(
        config_path=str(config_path) if config_path.exists() else None,
        camera_config_path=str(camera_config_path) if camera_config_path.exists() else None,
        show_camera=True
    )
    
    if mirror.setup():
        mirror.run()


if __name__ == "__main__":
    main()
