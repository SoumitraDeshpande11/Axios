#!/usr/bin/env python3
"""
Test script to visualize the humanoid robot in PyBullet.
Run this to see the robot and verify everything works.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pybullet as p
import pybullet_data
from robot_descriptions import simple_humanoid_description


def main():
    print("=" * 50)
    print("AXIOS - Robot Visualization Test")
    print("=" * 50)
    
    # Connect to PyBullet GUI
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
    # Load humanoid robot
    urdf_path = simple_humanoid_description.URDF_PATH
    print(f"Loading robot from: {urdf_path}")
    
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 1.0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False
    )
    
    # Print joint info
    num_joints = p.getNumJoints(robot_id)
    print(f"\nRobot loaded with {num_joints} joints:")
    print("-" * 40)
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        joint_type = info[2]
        type_name = {0: "REVOLUTE", 1: "PRISMATIC", 2: "SPHERICAL", 3: "PLANAR", 4: "FIXED"}.get(joint_type, "UNKNOWN")
        print(f"  Joint {i}: {name} ({type_name})")
    
    print("-" * 40)
    print("\nControls:")
    print("  - Use mouse to rotate camera")
    print("  - Scroll to zoom")
    print("  - Press Ctrl+C to exit")
    print("\nRunning simulation...")
    
    # Set camera position
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 1.0]
    )
    
    # Run simulation
    try:
        t = 0
        while True:
            # Simple arm movement demo
            left_shoulder_pitch = 0.5 * (1 + abs((t % 4) - 2) - 1)
            right_shoulder_pitch = 0.5 * (1 + abs(((t + 2) % 4) - 2) - 1)
            
            # Find and control arm joints
            for i in range(num_joints):
                info = p.getJointInfo(robot_id, i)
                name = info[1].decode("utf-8").lower()
                
                if "left" in name and "shoulder" in name and "pitch" in name:
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                           targetPosition=-left_shoulder_pitch, force=100)
                elif "right" in name and "shoulder" in name and "pitch" in name:
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                                           targetPosition=-right_shoulder_pitch, force=100)
            
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            t += 1.0 / 240.0
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
