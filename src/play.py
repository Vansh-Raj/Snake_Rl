import argparse
import os
import torch
import time

from src.config import *
from src.env.snake_env import SnakeEnv
from src.rl.dqn import DQNAgent, DQNConfig

def main(model_path: str):
    device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    env = SnakeEnv(render=True, play_mode=True)

    cfg = DQNConfig(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        lr=LR,
        gamma=GAMMA,
        device=device
    )
    agent = DQNAgent(cfg)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    agent.load(model_path)
    print(f"Loaded model: {model_path}")

    while True:
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action = agent.act(s_tensor)  # strictly greedy
            state, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Score: {info.get('score', 0)} | Episode reward: {total_reward:.2f}")
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.path.join(SAVE_DIR, BEST_MODEL_FILENAME))
    args = parser.parse_args()
    main(args.model_path)
