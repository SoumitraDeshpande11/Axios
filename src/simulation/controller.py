import numpy as np


class PDController:
    
    def __init__(self, num_joints, kp=100.0, kd=10.0, max_force=200.0):
        self.num_joints = num_joints
        self.kp = np.full(num_joints, kp)
        self.kd = np.full(num_joints, kd)
        self.max_force = np.full(num_joints, max_force)
        self.targets = np.zeros(num_joints)
        self.prev_error = np.zeros(num_joints)
        
    def set_gains(self, joint_idx, kp=None, kd=None, max_force=None):
        if kp is not None:
            self.kp[joint_idx] = kp
        if kd is not None:
            self.kd[joint_idx] = kd
        if max_force is not None:
            self.max_force[joint_idx] = max_force
    
    def set_targets(self, targets):
        self.targets = np.array(targets)
    
    def compute(self, current_positions, current_velocities, dt):
        error = self.targets - current_positions
        error_derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros_like(error)
        self.prev_error = error.copy()
        torques = self.kp * error + self.kd * (0 - current_velocities)
        torques = np.clip(torques, -self.max_force, self.max_force)
        return torques
    
    def get_forces(self):
        return self.max_force.tolist()


class TrajectoryInterpolator:
    
    def __init__(self, num_joints, max_velocity=5.0):
        self.num_joints = num_joints
        self.max_velocity = max_velocity
        self.current = np.zeros(num_joints)
        self.target = np.zeros(num_joints)
        
    def set_target(self, target):
        self.target = np.array(target)
    
    def update(self, dt):
        diff = self.target - self.current
        max_step = self.max_velocity * dt
        step = np.clip(diff, -max_step, max_step)
        self.current = self.current + step
        return self.current.copy()
    
    def reset(self, positions):
        self.current = np.array(positions)
        self.target = np.array(positions)
