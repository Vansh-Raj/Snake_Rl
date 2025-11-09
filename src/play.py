import argparse
import os
import sys
import torch
import time
import numpy as np

# Force pygame to use dummy driver in headless environments
os.environ["SDL_VIDEODRIVER"] = "dummy"

from src.config import *
from src.env.snake_env import SnakeEnv
from src.rl.dqn import DQNAgent, DQNConfig

try:
    import pygame
    import cv2
    from IPython.display import Video, display
except ImportError:
    pygame = None
    cv2 = None


def save_video(frames, output_path):
    """Safely save collected frames to an MP4 file."""
    if not frames or cv2 is None:
        print("⚠️ No frames to save or OpenCV not available.")
        return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"🎬 Gameplay video saved as: {output_path}")
    if "IPython" in sys.modules:
        display(Video(output_path, embed=True))


def main(model_path: str, record=False, output_path="gameplay.mp4", max_time=20):
    """Play the trained Snake game model for a fixed duration with video recording."""
    device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"

    # Always use headless rendering in Colab
    env = SnakeEnv(render=False, play_mode=True)

    # Initialize and load agent
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
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")
    agent.load(model_path)
    print(f"✅ Loaded trained model from: {model_path}")
    print(f"⏱️ Running gameplay for {max_time} seconds...")

    frames = []
    if record and cv2 is not None:
        print("🎥 Recording gameplay...")

    state = env.reset()
    done = False
    total_reward = 0.0
    start_time = time.time()

    try:
        while time.time() - start_time < max_time:
            # Restart if the snake dies
            if done:
                state = env.reset()
                done = False

            s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action = agent.act(s_tensor)
            state, reward, done, info = env.step(action)
            total_reward += reward

            # Force off-screen render
            if record and cv2 is not None and pygame is not None:
                surface = pygame.Surface((env.grid * CELL_PIXELS, env.grid * CELL_PIXELS))
                env.screen = surface
                env._render()
                frame = pygame.surfarray.array3d(surface)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)

        print(f"🏁 Final Score: {info.get('score', 0)} | Total Reward: {total_reward:.2f}")

        if record and frames:
            save_video(frames, output_path)

    except Exception as e:
        print(f"⚠️ Exception occurred: {e}")
        if record and frames:
            save_video(frames, output_path)
    finally:
        if pygame is not None:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play trained Snake RL model and record gameplay.")
    parser.add_argument("--model_path", type=str, default=os.path.join(SAVE_DIR, BEST_MODEL_FILENAME),
                        help="Path to the trained model file (.pth)")
    parser.add_argument("--record", action="store_true", help="Enable video recording (for Colab).")
    parser.add_argument("--output_path", type=str, default="gameplay.mp4", help="Output video filename.")
    parser.add_argument("--max_time", type=int, default=20, help="Gameplay duration in seconds.")
    args = parser.parse_args()

    main(args.model_path, record=args.record, output_path=args.output_path, max_time=args.max_time)
