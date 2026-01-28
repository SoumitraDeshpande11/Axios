import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from src.simulation.environment import AxiosEnv


def main():
    config_path = Path(__file__).parent.parent / "config" / "humanoid.yaml"
    env = AxiosEnv(config_path=str(config_path), render=True)
    env.connect()
    
    print("Axios Manual Control")
    print("Joint mapping:")
    for name, idx in env.humanoid.joint_name_to_index.items():
        print(f"  {idx}: {name}")
    
    print("\nUpperbody joints:")
    for name in env.humanoid.UPPER_BODY_JOINTS:
        idx = env.humanoid.get_joint_index(name)
        if idx >= 0:
            print(f"  {name}: index {idx}")
    
    targets = np.zeros(len(env.humanoid.UPPER_BODY_JOINTS))
    
    print("\nRunning simulation loop...")
    print("Press Ctrl+C to exit")
    
    try:
        t = 0
        while True:
            targets[0] = 0.5 * np.sin(t * 2)
            targets[3] = 0.5 * np.sin(t * 2 + np.pi)
            targets[2] = -0.5 - 0.3 * np.sin(t * 3)
            targets[5] = 0.5 + 0.3 * np.sin(t * 3)
            
            env.set_joint_targets(targets)
            env.step(8)
            
            t += 1.0 / 30.0
            time.sleep(1.0 / 30.0)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()


if __name__ == "__main__":
    main()
