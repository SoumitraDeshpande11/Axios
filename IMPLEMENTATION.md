# Axios Implementation Guide

Detailed technical specification for the Axios humanoid boxing robot project.

---

## Table of Contents

1. System Architecture
2. Robot Model Specification
3. Simulation Environment
4. Perception Pipeline
5. Motion Retargeting
6. ATOM Mirror System
7. RL Training System
8. Data Formats
9. API Reference

---

## 1. System Architecture

### High-Level Pipeline

```
+----------------+     +------------------+     +-------------------+
|                |     |                  |     |                   |
|  Camera Feed   +---->+  Pose Estimator  +---->+  Pose Filter      |
|  (OpenCV)      |     |  (MediaPipe)     |     |  (Kalman/1Euro)   |
|                |     |                  |     |                   |
+----------------+     +------------------+     +--------+----------+
                                                         |
                                                         v
+----------------+     +------------------+     +-------------------+
|                |     |                  |     |                   |
|  PyBullet Sim  +<----+  PD Controller   +<----+  Retargeting      |
|  (Physics)     |     |  (Joint Control) |     |  (Angle Mapping)  |
|                |     |                  |     |                   |
+----------------+     +------------------+     +-------------------+
```

### Data Flow Rates

| Stage | Frequency | Latency Budget |
|-------|-----------|----------------|
| Camera Capture | 30 Hz | 33 ms |
| Pose Estimation | 30 Hz | 20 ms |
| Filtering | 30 Hz | 1 ms |
| Retargeting | 30 Hz | 1 ms |
| Control Command | 30 Hz | 1 ms |
| Physics Step | 240 Hz | 4 ms |

Total target latency: Under 60 ms end-to-end.

---

## 2. Robot Model Specification

### Overview

We build a custom 8-part humanoid URDF optimized for boxing. This gives us full control over joint limits, masses, and proportions rather than using a generic model like NAO which has too many joints.

### Kinematic Tree

```
world (fixed)
  |
  +-- torso (base link)
        |
        +-- head
        |     (yaw joint)
        |
        +-- left_upper_arm
        |     (shoulder_pitch, shoulder_roll)
        |     |
        |     +-- left_lower_arm
        |           (elbow joint)
        |
        +-- right_upper_arm
        |     (shoulder_pitch, shoulder_roll)
        |     |
        |     +-- right_lower_arm
        |           (elbow joint)
        |
        +-- left_leg
        |     (hip joint)
        |
        +-- right_leg
              (hip joint)
```

### Link Specifications

#### Torso
- Shape: Box
- Dimensions: 0.35m (width) x 0.25m (depth) x 0.50m (height)
- Mass: 25.0 kg
- Position: Origin at center of mass
- Color: RGB(0.3, 0.3, 0.35)

#### Head
- Shape: Sphere
- Radius: 0.10m
- Mass: 4.5 kg
- Position: 0.35m above torso center
- Color: RGB(0.8, 0.7, 0.6)

#### Upper Arm (left/right)
- Shape: Capsule
- Radius: 0.045m
- Length: 0.28m
- Mass: 2.5 kg
- Position: Attached at shoulder (top corners of torso)
- Color: RGB(0.25, 0.25, 0.3)

#### Lower Arm (left/right)
- Shape: Capsule
- Radius: 0.035m
- Length: 0.25m
- Mass: 1.8 kg
- Position: Attached at elbow (end of upper arm)
- Includes fist collision sphere at end (radius 0.05m)
- Color: RGB(0.25, 0.25, 0.3)

#### Leg (left/right)
- Shape: Capsule
- Radius: 0.08m
- Length: 0.75m
- Mass: 10.0 kg
- Position: Attached at hip (bottom of torso)
- Color: RGB(0.2, 0.2, 0.25)

### Joint Specifications

| Joint Name | Type | Parent | Child | Axis | Lower (rad) | Upper (rad) | Max Vel (rad/s) | Max Force (Nm) |
|------------|------|--------|-------|------|-------------|-------------|-----------------|----------------|
| head_yaw | revolute | torso | head | 0 0 1 | -1.57 | 1.57 | 3.0 | 50 |
| left_shoulder_pitch | revolute | torso | left_upper_arm | 0 1 0 | -3.14 | 0.52 | 5.0 | 200 |
| left_shoulder_roll | revolute | left_upper_arm | left_upper_arm_roll | 1 0 0 | -0.52 | 2.62 | 5.0 | 200 |
| left_elbow | revolute | left_upper_arm_roll | left_lower_arm | 0 1 0 | -2.36 | 0.0 | 6.0 | 150 |
| right_shoulder_pitch | revolute | torso | right_upper_arm | 0 1 0 | -3.14 | 0.52 | 5.0 | 200 |
| right_shoulder_roll | revolute | right_upper_arm | right_upper_arm_roll | 1 0 0 | -2.62 | 0.52 | 5.0 | 200 |
| right_elbow | revolute | right_upper_arm_roll | right_lower_arm | 0 1 0 | 0.0 | 2.36 | 6.0 | 150 |
| left_hip | revolute | torso | left_leg | 0 1 0 | -0.79 | 0.79 | 2.0 | 100 |
| right_hip | revolute | torso | right_leg | 0 1 0 | -0.79 | 0.79 | 2.0 | 100 |

### Inertia Calculation

For each link, inertia tensor is computed from geometry:

Box (torso):
```
Ixx = (1/12) * m * (h^2 + d^2)
Iyy = (1/12) * m * (w^2 + d^2)
Izz = (1/12) * m * (w^2 + h^2)
```

Capsule (arms, legs):
```
Ixx = Iyy = (1/12) * m * L^2 + (1/4) * m * r^2
Izz = (1/2) * m * r^2
```

Sphere (head):
```
Ixx = Iyy = Izz = (2/5) * m * r^2
```

### Collision Groups

| Group | Links | Can Collide With |
|-------|-------|------------------|
| 0 | torso, head | opponent, ground |
| 1 | left_upper_arm, left_lower_arm | opponent, ground |
| 2 | right_upper_arm, right_lower_arm | opponent, ground |
| 3 | left_leg, right_leg | ground |

Self-collision disabled between adjacent links.

---

## 3. Simulation Environment

### PyBullet Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Time Step | 1/240 s | Standard for stable contact |
| Gravity | (0, 0, -9.81) | Earth gravity |
| Solver Iterations | 50 | Improved constraint solving |
| Contact ERP | 0.9 | Contact error reduction |
| Contact CFM | 0.0 | Contact constraint force mixing |
| Friction | 1.0 | Rubber-like ground |

### Ground Plane

- Infinite plane at z=0
- High friction (1.0)
- No restitution (0.0)

### Camera Setup (Visualization)

| Parameter | Value |
|-----------|-------|
| Distance | 2.0 m |
| Yaw | 0 deg |
| Pitch | -20 deg |
| Target | Robot torso |
| FOV | 60 deg |
| Resolution | 1280 x 720 |

### Environment API

```
class AxiosEnv:
    
    def __init__(self, render_mode, config_path):
        ...
    
    def reset(self):
        returns: observation dict
    
    def step(self, action):
        returns: observation, reward, terminated, truncated, info
    
    def render(self):
        returns: RGB array if render_mode is rgb_array
    
    def close(self):
        returns: None
    
    def get_state(self):
        returns: dict with joint positions, velocities, link states
    
    def set_joint_targets(self, targets):
        returns: None
```

### Observation Dictionary

```
{
    "joint_positions": np.array shape (9,),
    "joint_velocities": np.array shape (9,),
    "torso_orientation": np.array shape (4,) quaternion,
    "left_fist_position": np.array shape (3,),
    "right_fist_position": np.array shape (3,),
    "left_fist_velocity": np.array shape (3,),
    "right_fist_velocity": np.array shape (3,),
}
```

---

## 4. Perception Pipeline

### Camera Capture

Uses OpenCV VideoCapture with V4L2 backend on Linux, AVFoundation on macOS.

| Parameter | Default | Range |
|-----------|---------|-------|
| Device Index | 0 | 0-9 |
| Width | 640 | 320-1920 |
| Height | 480 | 240-1080 |
| FPS | 30 | 15-60 |
| Buffer Size | 1 | 1-5 |

Buffer size of 1 ensures we always get the latest frame with no lag.

### MediaPipe Pose

Configuration:

| Parameter | Value |
|-----------|-------|
| static_image_mode | False |
| model_complexity | 1 |
| smooth_landmarks | True |
| enable_segmentation | False |
| min_detection_confidence | 0.5 |
| min_tracking_confidence | 0.5 |

### Landmark Indices

Key landmarks for boxing:

| Index | Name | Use |
|-------|------|-----|
| 0 | nose | Head orientation |
| 11 | left_shoulder | Arm base |
| 12 | right_shoulder | Arm base |
| 13 | left_elbow | Arm joint |
| 14 | right_elbow | Arm joint |
| 15 | left_wrist | Fist position |
| 16 | right_wrist | Fist position |
| 23 | left_hip | Torso base |
| 24 | right_hip | Torso base |

### Coordinate System

MediaPipe output:
- x: 0 to 1, left to right in image
- y: 0 to 1, top to bottom in image
- z: Depth relative to hips, negative is closer to camera

Transformation to world frame:
```
world_x = -(mp_x - 0.5) * scale
world_y = mp_z * scale
world_z = -(mp_y - 0.5) * scale
```

Where scale is calibrated based on user distance (default 2.0).

### Kalman Filter

State vector per coordinate: [position, velocity]

Matrices:
```
F = [[1, dt],
     [0, 1]]

H = [[1, 0]]

Q = [[0.1, 0],
     [0, 0.1]]

R = [[0.5]]
```

Process noise Q tuned for smooth motion.
Measurement noise R tuned for MediaPipe jitter.

### One Euro Filter (Alternative)

Parameters:
| Name | Value |
|------|-------|
| min_cutoff | 1.0 |
| beta | 0.007 |
| d_cutoff | 1.0 |

Lower min_cutoff = more smoothing.
Higher beta = less lag during fast motion.

---

## 5. Motion Retargeting

### Algorithm Overview

1. Extract vectors from pose landmarks
2. Compute angles in human frame
3. Transform to robot frame
4. Apply calibration offsets
5. Clamp to joint limits
6. Apply rate limiting

### Vector Extraction

```
left_upper_arm_vec = left_elbow - left_shoulder
left_lower_arm_vec = left_wrist - left_elbow
right_upper_arm_vec = right_elbow - right_shoulder
right_lower_arm_vec = right_wrist - right_elbow
torso_vec = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
```

### Angle Computation

Shoulder pitch (forward/backward rotation):
```
pitch = atan2(-upper_arm_vec.z, -upper_arm_vec.y)
```

Shoulder roll (lateral rotation):
```
roll = atan2(upper_arm_vec.x, sqrt(upper_arm_vec.y^2 + upper_arm_vec.z^2))
```

Elbow flexion:
```
dot = upper_arm_vec dot lower_arm_vec
cross = upper_arm_vec cross lower_arm_vec
angle = atan2(norm(cross), dot)
elbow = pi - angle
```

### Calibration

T-pose calibration capture:
1. User stands in T-pose
2. Record landmark positions
3. Compute offset angles
4. Apply as correction to live poses

### Rate Limiting

Maximum joint velocity: 5 rad/s (for safety and smoothness)

```
delta = target - current
max_delta = max_velocity * dt
clamped_delta = clamp(delta, -max_delta, max_delta)
output = current + clamped_delta
```

### Mirroring

Robot mirrors human, so left human arm maps to right robot arm:
- left_shoulder landmarks -> right_shoulder joints
- right_shoulder landmarks -> left_shoulder joints
- x coordinates are negated

---

## 6. ATOM Mirror System

### Control Loop

```
initialize:
    camera = Camera(config)
    estimator = PoseEstimator(config)
    filter = PoseFilter(config)
    mapper = RetargetingMapper(config)
    env = AxiosEnv(config)
    controller = PDController(config)

loop at 30Hz:
    timestamp = current_time()
    
    frame = camera.get_frame()
    if frame is None:
        continue
    
    landmarks = estimator.estimate(frame)
    if landmarks is None:
        continue
    
    filtered = filter.update(landmarks, timestamp)
    
    joint_targets = mapper.compute_angles(filtered)
    
    controller.set_targets(joint_targets)
    
    for i in range(8):
        env.step(controller.get_commands())
    
    if recording:
        save_frame(env.render(), timestamp, joint_targets)
```

### Timing Management

Target loop time: 33.3 ms (30 Hz)

```
loop_start = time()
... do work ...
elapsed = time() - loop_start
sleep_time = max(0, target_period - elapsed)
sleep(sleep_time)
```

### Recording Format

Demonstration data saved as:
```
recordings/
    session_YYYYMMDD_HHMMSS/
        metadata.yaml
        frames/
            000000.jpg
            000001.jpg
            ...
        states.npz
```

metadata.yaml:
```
start_time: ISO timestamp
duration_seconds: float
frame_count: int
fps: 30
config: dict
```

states.npz:
```
timestamps: shape (N,)
joint_positions: shape (N, 9)
joint_velocities: shape (N, 9)
joint_targets: shape (N, 9)
landmarks: shape (N, 33, 3)
```

---

## 7. RL Training System

### Environment Wrapper

Gymnasium-compatible interface.

Observation Space:
```
Box(
    low=-inf,
    high=inf,
    shape=(29,),
    dtype=float32
)

Layout:
[0:9]   - joint positions
[9:18]  - joint velocities
[18:21] - left fist position (relative to torso)
[21:24] - right fist position (relative to torso)
[24:27] - opponent position (relative to torso)
[27:29] - opponent guard state (left, right)
```

Action Space:
```
Box(
    low=-0.1,
    high=0.1,
    shape=(7,),
    dtype=float32
)

Layout:
[0] - left_shoulder_pitch delta
[1] - left_shoulder_roll delta
[2] - left_elbow delta
[3] - right_shoulder_pitch delta
[4] - right_shoulder_roll delta
[5] - right_elbow delta
[6] - head_yaw delta
```

### Reward Function

```
def compute_reward(state, action, next_state, info):
    reward = 0.0
    
    if info["landed_hit"]:
        if info["hit_location"] == "head":
            reward += 10.0
        elif info["hit_location"] == "body":
            reward += 5.0
    
    if info["received_hit"]:
        if info["hit_location"] == "head":
            reward -= 8.0
        elif info["hit_location"] == "body":
            reward -= 3.0
    
    guard_bonus = compute_guard_score(state)
    reward += 0.1 * guard_bonus
    
    extension_bonus = compute_extension_score(state, opponent_pos)
    reward += 0.05 * extension_bonus
    
    energy_cost = sum(action ** 2)
    reward -= 0.01 * energy_cost
    
    if abs(state["torso_pitch"]) > 0.3:
        reward -= 0.5
    
    return reward
```

### Opponent

Phase 1: Static dummy
- Fixed position 1.5m in front
- No movement
- Hitboxes only

Phase 2: Scripted opponent
- Random jabs at intervals
- Predictable patterns
- Teaches defense

Phase 3: Self-play
- Clone of trained policy
- Updated periodically
- Emergent strategies

### Training Configuration

PPO Hyperparameters:
| Parameter | Value |
|-----------|-------|
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

Policy Network:
```
MLP:
    Input: 29
    Hidden: [256, 256]
    Activation: Tanh
    Output: 7 (mean) + 7 (log_std)
```

Value Network:
```
MLP:
    Input: 29
    Hidden: [256, 256]
    Activation: Tanh
    Output: 1
```

### Curriculum Learning

Stage 1 (0-1M steps):
- Opponent stationary
- Large hit targets
- High reward for any extension

Stage 2 (1M-3M steps):
- Opponent guards randomly
- Normal hit targets
- Balanced rewards

Stage 3 (3M-10M steps):
- Opponent fights back (scripted)
- Defense matters
- Full reward function

---

## 8. Data Formats

### Configuration Files (YAML)

humanoid.yaml:
```
joints:
  head_yaw:
    kp: 50.0
    kd: 5.0
    max_force: 50.0
  left_shoulder_pitch:
    kp: 200.0
    kd: 20.0
    max_force: 200.0
  ...

default_pose:
  head_yaw: 0.0
  left_shoulder_pitch: -0.5
  left_shoulder_roll: 0.3
  left_elbow: -0.8
  right_shoulder_pitch: -0.5
  right_shoulder_roll: -0.3
  right_elbow: 0.8
  left_hip: 0.0
  right_hip: 0.0
```

camera.yaml:
```
device_index: 0
width: 640
height: 480
fps: 30

filter:
  type: kalman
  process_noise: 0.1
  measurement_noise: 0.5
```

training.yaml:
```
algorithm: PPO
total_timesteps: 10000000
n_envs: 8
checkpoint_freq: 100000

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
```

### Checkpoint Format

Saved by Stable-Baselines3:
```
checkpoints/
    ppo_boxing_1000000_steps.zip
```

Contains:
- Policy network weights
- Value network weights
- Optimizer state
- Normalization statistics

---

## 9. API Reference

### AxiosEnv

```
class AxiosEnv:
    
    def __init__(
        self,
        render_mode: str = None,
        config_path: str = "config/humanoid.yaml"
    ):
        ...
    
    def reset(
        self,
        seed: int = None,
        options: dict = None
    ) -> tuple[dict, dict]:
        ...
    
    def step(
        self,
        action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        ...
    
    def render(self) -> np.ndarray:
        ...
    
    def close(self) -> None:
        ...
```

### Humanoid

```
class Humanoid:
    
    def __init__(
        self,
        physics_client: int,
        urdf_path: str,
        base_position: tuple = (0, 0, 1)
    ):
        ...
    
    def get_joint_positions(self) -> np.ndarray:
        ...
    
    def get_joint_velocities(self) -> np.ndarray:
        ...
    
    def set_joint_targets(
        self,
        targets: np.ndarray,
        gains: np.ndarray = None
    ) -> None:
        ...
    
    def get_link_state(
        self,
        link_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        ...
```

### PoseEstimator

```
class PoseEstimator:
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        ...
    
    def estimate(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        ...
    
    def close(self) -> None:
        ...
```

### RetargetingMapper

```
class RetargetingMapper:
    
    def __init__(
        self,
        joint_limits: dict,
        calibration: dict = None
    ):
        ...
    
    def calibrate(
        self,
        landmarks: np.ndarray
    ) -> None:
        ...
    
    def compute_angles(
        self,
        landmarks: np.ndarray
    ) -> np.ndarray:
        ...
```

### AtomMirror

```
class AtomMirror:
    
    def __init__(
        self,
        camera_config: str,
        humanoid_config: str,
        render: bool = True
    ):
        ...
    
    def run(
        self,
        record: bool = False,
        output_dir: str = None
    ) -> None:
        ...
    
    def stop(self) -> None:
        ...
```

---

## Appendix: Joint Index Mapping

| Index | Joint Name |
|-------|------------|
| 0 | head_yaw |
| 1 | left_shoulder_pitch |
| 2 | left_shoulder_roll |
| 3 | left_elbow |
| 4 | right_shoulder_pitch |
| 5 | right_shoulder_roll |
| 6 | right_elbow |
| 7 | left_hip |
| 8 | right_hip |
