import os
import math
import random
import numpy as np
import torch

from src.config import *
from src.env.snake_env import SnakeEnv
from src.rl.dqn import DQNAgent, DQNConfig
from src.rl.replay_buffer import ReplayBuffer

def linear_epsilon(step, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    return max(end, start - (start - end) * (step / decay_steps))

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    env = SnakeEnv(render=RENDER_TRAIN)
    cfg = DQNConfig(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        lr=LR,
        gamma=GAMMA,
        device=device
    )
    agent = DQNAgent(cfg)
    buffer = ReplayBuffer(BUFFER_CAPACITY, STATE_SIZE, device)

    global_step = 0
    best_score = -1

    for ep in range(1, MAX_EPISODES + 1):
        state = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        epsilon = linear_epsilon(global_step, EPS_START, EPS_END, EPS_DECAY_STEPS)

        while not done and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            global_step += 1

            s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE - 1)
            else:
                action = agent.act(s_tensor)

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            # Learn
            if len(buffer) >= max(BATCH_SIZE, LEARN_START):
                for _ in range(TRAIN_STEPS_PER_ENV_STEP):
                    batch = buffer.sample(BATCH_SIZE)
                    loss = agent.learn(batch)

            # Target update
            if global_step % TARGET_UPDATE_EVERY == 0:
                agent.update_target()

            state = next_state
            ep_reward += reward

        score = info.get("score", 0)
        print(f"Episode {ep:4d} | steps: {steps:4d} | score: {score:3d} | reward: {ep_reward:7.2f} | eps: {epsilon:5.3f}")

        # Save periodic
        if ep % SAVE_EVERY_EPISODES == 0:
            path = os.path.join(SAVE_DIR, LAST_MODEL_FILENAME)
            agent.save(path)
            print(f"Saved: {path}")

        # Save best by score
        if score > best_score:
            best_score = score
            path = os.path.join(SAVE_DIR, BEST_MODEL_FILENAME)
            agent.save(path)
            print(f"New best score {best_score}. Saved: {path}")

    # final save
    path = os.path.join(SAVE_DIR, LAST_MODEL_FILENAME)
    agent.save(path)
    print(f"Training done. Saved last: {path}")

if __name__ == "__main__":
    main()
