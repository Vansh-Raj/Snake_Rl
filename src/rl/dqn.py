import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DQNConfig:
    state_size: int
    action_size: int
    hidden_size: int
    lr: float
    gamma: float
    device: str

class QNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.q = QNetwork(cfg.state_size, cfg.hidden_size, cfg.action_size).to(cfg.device)
        self.target = QNetwork(cfg.state_size, cfg.hidden_size, cfg.action_size).to(cfg.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    @torch.no_grad()
    def act(self, state_tensor):
        """Greedy action (no exploration); state_tensor: shape [state_size] on device."""
        q_values = self.q(state_tensor.unsqueeze(0))  # [1, A]
        action = torch.argmax(q_values, dim=1).item()
        return action

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch
        # shapes: [B, S], [B], [B], [B, S], [B]
        q_pred = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        loss = self.loss_fn(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save(self.q.state_dict(), path)

    def load(self, path: str):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target()
