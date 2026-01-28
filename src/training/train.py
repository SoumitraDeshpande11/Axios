import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import argparse
from datetime import datetime

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("stable-baselines3 not installed. Run: pip install stable-baselines3")

from src.training.env_wrapper import BoxingEnv


def make_env(rank, seed=0):
    def _init():
        env = BoxingEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path, output_dir):
    if not SB3_AVAILABLE:
        print("Cannot train without stable-baselines3")
        return
    
    config = load_config(config_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_boxing_{timestamp}"
    run_dir = output_path / run_name
    run_dir.mkdir(exist_ok=True)
    
    n_envs = config.get("n_envs", 4)
    
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    eval_env = DummyVecEnv([make_env(0)])
    
    ppo_config = config.get("ppo", {})
    policy_config = config.get("policy", {})
    
    net_arch = policy_config.get("net_arch", [256, 256])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        n_steps=ppo_config.get("n_steps", 2048),
        batch_size=ppo_config.get("batch_size", 64),
        n_epochs=ppo_config.get("n_epochs", 10),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        policy_kwargs={"net_arch": net_arch},
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard")
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 50000) // n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_boxing"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=config.get("eval_freq", 10000) // n_envs,
        n_eval_episodes=config.get("eval_episodes", 5),
        deterministic=True,
        render=False
    )
    
    total_timesteps = config.get("total_timesteps", 1000000)
    
    print(f"Training for {total_timesteps} timesteps...")
    print(f"Output directory: {run_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    model.save(str(run_dir / "final_model"))
    print(f"Training complete. Model saved to {run_dir / 'final_model'}")
    
    env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/training.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/training",
        help="Output directory"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps"
    )
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent.parent / args.config
    output_dir = Path(__file__).parent.parent.parent / args.output
    
    if args.timesteps:
        config = load_config(config_path)
        config["total_timesteps"] = args.timesteps
        with open(config_path, "w") as f:
            yaml.dump(config, f)
    
    train(str(config_path), str(output_dir))


if __name__ == "__main__":
    main()
