import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs):
        return self.net(obs)
