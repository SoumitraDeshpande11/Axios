import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import yaml

from src.simulation.environment import AxiosEnv
from src.simulation.controller import TrajectoryInterpolator
from src.perception.camera import Camera
from src.perception.pose_estimator import PoseEstimator
from src.perception.filters import PoseFilter
from src.retargeting.mapper import RetargetingMapper


class AtomMirror:
    
    def __init__(self, config_path=None, camera_config_path=None):
        self.config_path = config_path
        self.camera_config = self._load_camera_config(camera_config_path)
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
                return yaml.safe_load(f)
        return {}
    
    def setup(self):
        self.env = AxiosEnv(config_path=self.config_path, render=True)
        self.env.connect()
        
        cam_cfg = self.camera_config
        self.camera = Camera(
            device_index=cam_cfg.get("device_index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30)
        )
        self.camera.open()
        
        pose_cfg = cam_cfg.get("pose_estimation", {})
        self.pose_estimator = PoseEstimator(
            model_complexity=pose_cfg.get("model_complexity", 1),
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5)
        )
        
        filter_cfg = cam_cfg.get("filter", {})
        self.pose_filter = PoseFilter(
            num_landmarks=33,
            num_coords=3,
            min_cutoff=filter_cfg.get("min_cutoff", 1.0),
            beta=filter_cfg.get("beta", 0.007)
        )
        
        self.mapper = RetargetingMapper(mirror=True)
        self.interpolator = TrajectoryInterpolator(num_joints=6, max_velocity=5.0)
        
    def run(self):
        self.running = True
        print("AtomMirror running. Press Ctrl+C to stop.")
        print("Stand in T-pose to calibrate...")
        
        calibrated = False
        calibration_frames = 0
        calibration_target = 30
        
        try:
            while self.running:
                loop_start = time.time()
                
                frame, timestamp = self.camera.get_frame_rgb()
                if frame is None:
                    continue
                
                landmarks = self.pose_estimator.estimate(frame)
                if landmarks is None:
                    self.env.step(8)
                    continue
                
                filtered = self.pose_filter.filter(landmarks, timestamp)
                boxing_landmarks = self._extract_boxing_landmarks(filtered)
                
                if not calibrated:
                    calibration_frames += 1
                    if calibration_frames >= calibration_target:
                        self.mapper.calibrate(boxing_landmarks)
                        self.interpolator.reset(np.zeros(6))
                        calibrated = True
                        print("Calibrated. Start moving.")
                    continue
                
                joint_angles = self.mapper.compute_angles(boxing_landmarks)
                if joint_angles is not None:
                    self.interpolator.set_target(joint_angles)
                    smoothed = self.interpolator.update(self.frame_time)
                    self.env.set_joint_targets(smoothed)
                
                self.env.step(8)
                
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.frame_time - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def _extract_boxing_landmarks(self, landmarks):
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
    
    def stop(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
        if self.pose_estimator is not None:
            self.pose_estimator.close()
        if self.env is not None:
            self.env.close()


def main():
    base = Path(__file__).parent.parent
    config_path = base / "config" / "humanoid.yaml"
    camera_config_path = base / "config" / "camera.yaml"
    
    mirror = AtomMirror(
        config_path=str(config_path),
        camera_config_path=str(camera_config_path)
    )
    mirror.setup()
    mirror.run()


if __name__ == "__main__":
    main()
