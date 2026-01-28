import pybullet as p
import pybullet_data
from robot_descriptions import simple_humanoid_description
import numpy as np
from pathlib import Path


class Humanoid:
    
    UPPER_BODY_JOINTS = [
        "left_shoulder_pitch",
        "left_shoulder_roll", 
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_elbow",
    ]
    
    def __init__(self, physics_client, base_position=(0, 0, 1.0)):
        self.client = physics_client
        self.base_position = base_position
        self.robot_id = None
        self.joint_name_to_index = {}
        self.joint_limits = {}
        self.num_joints = 0
        
    def load(self):
        urdf_path = simple_humanoid_description.URDF_PATH
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=self.base_position,
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
            physicsClientId=self.client
        )
        self._build_joint_map()
        return self.robot_id
        
    def _build_joint_map(self):
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            self.joint_name_to_index[joint_name] = i
            if joint_type != p.JOINT_FIXED:
                self.joint_limits[joint_name] = (lower_limit, upper_limit)
    
    def get_joint_index(self, joint_name):
        return self.joint_name_to_index.get(joint_name, -1)
    
    def get_joint_positions(self):
        positions = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.robot_id, i, physicsClientId=self.client)
            positions.append(joint_state[0])
        return np.array(positions)
    
    def get_joint_velocities(self):
        velocities = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.robot_id, i, physicsClientId=self.client)
            velocities.append(joint_state[1])
        return np.array(velocities)
    
    def get_upper_body_positions(self):
        positions = []
        for name in self.UPPER_BODY_JOINTS:
            idx = self.get_joint_index(name)
            if idx >= 0:
                state = p.getJointState(self.robot_id, idx, physicsClientId=self.client)
                positions.append(state[0])
        return np.array(positions)
    
    def set_joint_positions(self, joint_indices, positions, forces=None):
        if forces is None:
            forces = [100.0] * len(joint_indices)
        for idx, pos, force in zip(joint_indices, positions, forces):
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=pos,
                force=force,
                physicsClientId=self.client
            )
    
    def set_upper_body_positions(self, positions, forces=None):
        if forces is None:
            forces = [200.0] * len(positions)
        indices = []
        for name in self.UPPER_BODY_JOINTS:
            idx = self.get_joint_index(name)
            if idx >= 0:
                indices.append(idx)
        valid_count = min(len(indices), len(positions))
        self.set_joint_positions(indices[:valid_count], positions[:valid_count], forces[:valid_count])
    
    def get_link_state(self, link_index):
        state = p.getLinkState(self.robot_id, link_index, physicsClientId=self.client)
        position = np.array(state[0])
        orientation = np.array(state[1])
        return position, orientation
    
    def get_base_state(self):
        position, orientation = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        velocity, angular = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        return {
            "position": np.array(position),
            "orientation": np.array(orientation),
            "velocity": np.array(velocity),
            "angular_velocity": np.array(angular)
        }
    
    def reset_pose(self):
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0.0, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.robot_id,
            self.base_position,
            [0, 0, 0, 1],
            physicsClientId=self.client
        )
