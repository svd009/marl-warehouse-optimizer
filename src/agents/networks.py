import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkerActorNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return torch.distributions.Categorical(logits=logits)

class WorkerCriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        v = self.value(x)
        return v.squeeze(-1)

class CentralCriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.value(x)
        return v.squeeze(-1)
