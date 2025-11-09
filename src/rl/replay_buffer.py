import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, device: str):
        self.capacity = capacity
        self.device = device

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.int64)

        self.idx = 0
        self.full = False

    def push(self, s, a, r, s2, d):
        self.states[self.idx] = s
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_states[self.idx] = s2
        self.dones[self.idx] = int(d)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idxs = np.random.choice(n, batch_size, replace=False)
        states = torch.from_numpy(self.states[idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[idxs]).to(self.device)
        return states, actions, rewards, next_states, dones
