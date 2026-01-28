import pybullet as p
import pybullet_data
import numpy as np
import yaml
from pathlib import Path

from .humanoid import Humanoid


class AxiosEnv:
    
    def __init__(self, config_path=None, render=True):
        self.render = render
        self.config = self._load_config(config_path)
        self.client = None
        self.humanoid = None
        self.ground_id = None
        self.simulation_hz = self.config.get("simulation_hz", 240)
        self.time_step = 1.0 / self.simulation_hz
        
    def _load_config(self, config_path):
        if config_path is None:
            return {}
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}
    
    def connect(self):
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._setup_physics()
        self._load_ground()
        self._load_humanoid()
        return self.client
    
    def _setup_physics(self):
        physics = self.config.get("physics", {})
        gravity = physics.get("gravity", -9.81)
        p.setGravity(0, 0, gravity, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        solver_iterations = physics.get("solver_iterations", 50)
        p.setPhysicsEngineParameter(
            numSolverIterations=solver_iterations,
            physicsClientId=self.client
        )
    
    def _load_ground(self):
        self.ground_id = p.loadURDF(
            "plane.urdf",
            physicsClientId=self.client
        )
    
    def _load_humanoid(self):
        robot_config = self.config.get("robot", {})
        base_pos = robot_config.get("base_position", [0, 0, 1.0])
        self.humanoid = Humanoid(self.client, base_position=tuple(base_pos))
        self.humanoid.load()
    
    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation(physicsClientId=self.client)
    
    def get_state(self):
        return {
            "joint_positions": self.humanoid.get_joint_positions(),
            "joint_velocities": self.humanoid.get_joint_velocities(),
            "base": self.humanoid.get_base_state(),
            "upper_body_positions": self.humanoid.get_upper_body_positions()
        }
    
    def set_joint_targets(self, targets, forces=None):
        self.humanoid.set_upper_body_positions(targets, forces)
    
    def reset(self):
        self.humanoid.reset_pose()
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.client)
        return self.get_state()
    
    def render_frame(self):
        width = 640
        height = 480
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 1],
            distance=3.0,
            yaw=0,
            pitch=-20,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client
        )
        return np.array(rgb[:, :, :3])
    
    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
