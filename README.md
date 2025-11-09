#  Reinforcement Learning Snake Game

An intelligent **Snake Game AI** built using **Deep Q-Networks (DQN)** — a deep reinforcement learning algorithm that enables an agent to learn how to play the classic Snake game autonomously.  
This project demonstrates how an AI agent can learn from trial and error to maximize rewards through gameplay, evolving from random actions to optimized decision-making.

##  Project Overview

The **Reinforcement Learning Snake Game** aims to simulate autonomous learning behavior using **Deep Reinforcement Learning (DRL)**.  
Unlike traditional rule-based AI, this agent uses **Q-learning** with a **neural network function approximator** to learn strategies for survival and score maximization.

###  Core Objectives
- Train an agent to play Snake using **Deep Q-Network (DQN)**.
- Implement experience replay and target network stabilization.
- Visualize learning progression and agent gameplay.
- Demonstrate reinforcement learning in a dynamic, grid-based environment.

##  Technologies Used

| Component | Description |
|------------|-------------|
| **Language** | Python 3.11+ |
| **Frameworks** | PyTorch, Pygame |
| **Visualization** | Matplotlib, OpenCV |
| **Environment** | Custom-built Snake environment |
| **Training** | Deep Q-Learning with experience replay |

##  Project Structure

```
Snake_RL/
│
├── src/
│   ├── config.py                 # Global constants and training parameters
│   ├── env/
│   │   └── snake_env.py          # Game environment and reward logic
│   ├── rl/
│   │   ├── dqn.py                # Deep Q-Network agent implementation
│   │   └── replay_buffer.py      # Experience memory for training
│   ├── train.py                  # Main training script
│   └── play.py                   # Gameplay and video recording
│
├── checkpoints/                  # Saved model weights (.pth)
├── gameplay.mp4                  # Example recorded gameplay (generated after play)
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

##  Setup & Installation

###  Local Setup
```bash
git clone https://github.com/<your-username>/snake-rl.git
cd snake-rl
pip install -r requirements.txt
```

### ☁️ Run on Google Colab
If using Google Colab:
```bash
!git clone https://github.com/<your-username>/snake-rl.git
%cd snake-rl
!pip install pygame torch numpy opencv-python
```

##  Training the Agent

Run the training script:
```bash
python -m src.train
```

This will:
- Initialize the Snake environment  
- Train the DQN agent  
- Save the best-performing model in `/checkpoints`  

To adjust hyperparameters, edit values in `src/config.py`.

##  Playing the Trained Model

###  Run Locally
```bash
python -m src.play --model_path checkpoints/best_model.pth
```

###  Run & Record in Colab
```bash
!python -m src.play --record --model_path checkpoints/best_model.pth --max_time 20
from IPython.display import Video
Video("gameplay.mp4", embed=True)
```

##  Results

| Metric | Observation |
|--------|-------------|
| **Average Score (Trained)** | 35–45 |
| **Average Score (Untrained)** | 1–3 |
| **Training Episodes** | 500–1000 |
| **Reward Trend** | Increasing, converging after ~400 episodes |

##  Future Enhancements

- Implement **Double DQN** and **Dueling DQN** architectures for improved stability.  
- Add **CNN-based visual state representation** (learning from pixels).  
- Introduce **multi-agent competition or cooperation** modes.  
- Develop a **web dashboard** for real-time performance monitoring.  
- Extend to **3D simulation** or robotic pathfinding.

##  References

- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529–533.  
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.  
- PyTorch (2024). *PyTorch Documentation.* https://pytorch.org/  
- Pygame (2024). *Python Game Development Library.* https://www.pygame.org/  
- OpenAI (2018). *OpenAI Baselines: Reinforcement Learning Implementations.* https://github.com/openai/baselines  

## 🏁 Acknowledgements
Developed as part of an academic project exploring **Deep Reinforcement Learning applications** in autonomous decision systems.  
This project illustrates the potential of AI to learn complex behaviors from experience — a foundational step toward adaptive, intelligent control systems.

🧠 *"The more the agent plays, the smarter it gets — not because it was told how, but because it learned why."*
