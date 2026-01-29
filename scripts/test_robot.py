#!/usr/bin/env python3
"""
Test script to visualize the humanoid robot in PyBullet.
Uses the correct Simple Humanoid URDF with proper joint names.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pybullet as p
import pybullet_data
from robot_descriptions import simple_humanoid_description
import math


def main():
    print("=" * 50)
    print("AXIOS - Robot Visualization Test")
    print("=" * 50)
    
    # Connect to PyBullet GUI
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
    # Load humanoid robot with FIXED BASE
    urdf_path = simple_humanoid_description.URDF_PATH
    print(f"Loading robot from: {urdf_path}")
    
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.85],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True  # Fixed base - robot won't fall
    )
    
    # Get the upper body joint indices
    arm_joints = {
        "LARM_SHOULDER_P": None,
        "LARM_SHOULDER_R": None,
        "LARM_ELBOW": None,
        "RARM_SHOULDER_P": None,
        "RARM_SHOULDER_R": None,
        "RARM_ELBOW": None,
    }
    
    num_joints = p.getNumJoints(robot_id)
    print(f"\nRobot loaded with {num_joints} joints")
    print("-" * 40)
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        if name in arm_joints:
            arm_joints[name] = i
            print(f"  Found: {name} at index {i}")
    
    print("-" * 40)
    print("\nControls:")
    print("  - Use mouse to rotate camera")
    print("  - Scroll to zoom")
    print("  - Robot arms will move automatically")
    print("  - Press Ctrl+C to exit")
    print("\nRunning simulation...")
    
    # Set camera position
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=30,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0.9]
    )
    
    # Run simulation with arm movements
    try:
        t = 0
        while True:
            # Boxing-like arm movements
            # Left jab
            left_shoulder_p = -0.8 + 0.5 * math.sin(t * 3)
            left_shoulder_r = 0.3
            left_elbow = -1.0 - 0.5 * math.sin(t * 3)
            
            # Right cross (offset timing)
            right_shoulder_p = -0.8 + 0.5 * math.sin(t * 3 + math.pi)
            right_shoulder_r = 0.3
            right_elbow = -1.0 - 0.5 * math.sin(t * 3 + math.pi)
            
            # Apply joint controls
            if arm_joints["LARM_SHOULDER_P"] is not None:
                p.setJointMotorControl2(robot_id, arm_joints["LARM_SHOULDER_P"], 
                                       p.POSITION_CONTROL, targetPosition=left_shoulder_p, force=150)
            if arm_joints["LARM_SHOULDER_R"] is not None:
                p.setJointMotorControl2(robot_id, arm_joints["LARM_SHOULDER_R"],
                                       p.POSITION_CONTROL, targetPosition=left_shoulder_r, force=150)
            if arm_joints["LARM_ELBOW"] is not None:
                p.setJointMotorControl2(robot_id, arm_joints["LARM_ELBOW"],
                                       p.POSITION_CONTROL, targetPosition=left_elbow, force=100)
            
            if arm_joints["RARM_SHOULDER_P"] is not None:
                p.setJointMotorControl2(robot_id, arm_joints["RARM_SHOULDER_P"],
                                       p.POSITION_CONTROL, targetPosition=right_shoulder_p, force=150)
            if arm_joints["RARM_SHOULDER_R"] is not None:
                p.setJointMotorControl2(robot_id, arm_joints["RARM_SHOULDER_R"],
                                       p.POSITION_CONTROL, targetPosition=right_shoulder_r, force=150)
            if arm_joints["RARM_ELBOW"] is not None:
                p.setJointMotorControl2(robot_id, arm_joints["RARM_ELBOW"],
                                       p.POSITION_CONTROL, targetPosition=right_elbow, force=100)
            
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            t += 1.0 / 240.0
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
