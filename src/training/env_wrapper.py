import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from robot_descriptions import simple_humanoid_description


class BoxingEnv(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.client = None
        self.robot_id = None
        self.opponent_id = None
        self.ground_id = None
        
        self.simulation_hz = 240
        self.control_hz = 30
        self.steps_per_action = self.simulation_hz // self.control_hz
        self.time_step = 1.0 / self.simulation_hz
        self.max_episode_steps = 1000
        self.current_step = 0
        
        self.num_joints = 6
        self.joint_indices = []
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        self.joint_targets = np.zeros(self.num_joints)
        self.opponent_position = np.array([1.5, 0.0, 1.0])
        self.opponent_guard = np.array([0.5, 0.5])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.client is not None:
            p.disconnect(self.client)
        
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        self._load_robot()
        self._load_opponent()
        
        self.joint_targets = np.zeros(self.num_joints)
        self.current_step = 0
        
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.client)
        
        obs = self._get_observation()
        info = {}
        return obs, info
    
    def _load_robot(self):
        urdf_path = simple_humanoid_description.URDF_PATH
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 1.0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
            physicsClientId=self.client
        )
        
        joint_names = [
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_elbow",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_elbow",
        ]
        
        self.joint_indices = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            if name in joint_names:
                self.joint_indices.append(i)
    
    def _load_opponent(self):
        col_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=0.3,
            height=1.0,
            physicsClientId=self.client
        )
        vis_shape = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=0.3,
            length=1.0,
            rgbaColor=[0.8, 0.2, 0.2, 1.0],
            physicsClientId=self.client
        )
        self.opponent_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=self.opponent_position.tolist(),
            physicsClientId=self.client
        )
    
    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        self.joint_targets = self.joint_targets + action
        self.joint_targets = np.clip(self.joint_targets, -2.5, 2.5)
        
        for idx, target in zip(self.joint_indices, self.joint_targets):
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=200.0,
                physicsClientId=self.client
            )
        
        for _ in range(self.steps_per_action):
            p.stepSimulation(physicsClientId=self.client)
        
        self.current_step += 1
        
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        joint_positions = []
        joint_velocities = []
        for idx in self.joint_indices:
            state = p.getJointState(self.robot_id, idx, physicsClientId=self.client)
            joint_positions.append(state[0])
            joint_velocities.append(state[1])
        
        joint_positions = np.array(joint_positions, dtype=np.float32)
        joint_velocities = np.array(joint_velocities, dtype=np.float32)
        
        if len(joint_positions) < 6:
            joint_positions = np.pad(joint_positions, (0, 6 - len(joint_positions)))
            joint_velocities = np.pad(joint_velocities, (0, 6 - len(joint_velocities)))
        
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        
        left_fist = self._get_fist_position("left")
        right_fist = self._get_fist_position("right")
        
        relative_opponent = self.opponent_position - np.array(base_pos)
        
        obs = np.concatenate([
            joint_positions[:6],
            joint_velocities[:6],
            left_fist - np.array(base_pos),
            right_fist - np.array(base_pos),
            relative_opponent,
            self.opponent_guard,
            np.zeros(4)
        ]).astype(np.float32)
        
        return obs[:28]
    
    def _get_fist_position(self, side):
        target_names = {
            "left": ["left_wrist", "left_hand", "left_elbow"],
            "right": ["right_wrist", "right_hand", "right_elbow"]
        }
        
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            if any(t in name.lower() for t in target_names[side]):
                state = p.getLinkState(self.robot_id, i, physicsClientId=self.client)
                return np.array(state[0])
        
        base_pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        return np.array(base_pos)
    
    def _compute_reward(self):
        reward = 0.0
        
        left_fist = self._get_fist_position("left")
        right_fist = self._get_fist_position("right")
        
        left_dist = np.linalg.norm(left_fist - self.opponent_position)
        right_dist = np.linalg.norm(right_fist - self.opponent_position)
        
        if left_dist < 0.4:
            reward += 5.0
        if right_dist < 0.4:
            reward += 5.0
        
        reward += 0.1 * max(0, 2.0 - left_dist)
        reward += 0.1 * max(0, 2.0 - right_dist)
        
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        euler = p.getEulerFromQuaternion(base_orn)
        if abs(euler[0]) > 0.3 or abs(euler[1]) > 0.3:
            reward -= 1.0
        
        reward -= 0.01 * np.sum(np.abs(self.joint_targets))
        
        return float(reward)
    
    def _check_terminated(self):
        base_pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        if base_pos[2] < 0.5:
            return True
        return False
    
    def render(self):
        if self.render_mode == "rgb_array":
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0, 1],
                distance=3.0,
                yaw=45,
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
                physicsClientId=self.client
            )
            return np.array(rgb[:, :, :3], dtype=np.uint8)
        return None
    
    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
