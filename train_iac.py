import os
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# Add src/ to Python path
sys.path.insert(0, './src')

print("Loading environment...")
from env.warehouse_env import WarehouseEnv

# Define networks inline (no import needed)
class WorkerActorNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return torch.distributions.Categorical(logits=logits)

class WorkerCriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.value(x).squeeze(-1)

# Config class (no dataclass needed)
class Config:
    num_episodes = 50
    gamma = 0.99
    actor_lr = 3e-4
    critic_lr = 3e-4
    max_steps = 200

def make_env():
    return WarehouseEnv()

def main():
    cfg = Config()
    device = torch.device("cpu")
    
    print("✅ Creating environment...")
    env = make_env()
    
    # Your exact setup
    num_agents = 4
    obs_dim = 10 * 10 * 2  # 200
    n_actions = 6
    
    print(f"✅ Training {num_agents} agents | obs_dim={obs_dim} | actions={n_actions}")
    
    # Create networks + optimizers
    actors = [WorkerActorNet(obs_dim, n_actions).to(device) for _ in range(num_agents)]
    critics = [WorkerCriticNet(obs_dim).to(device) for _ in range(num_agents)]
    actor_opts = [optim.Adam(actor.parameters(), lr=cfg.actor_lr) for actor in actors]
    critic_opts = [optim.Adam(critic.parameters(), lr=cfg.critic_lr) for critic in critics]
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/iac.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("episode,total_reward,steps,picked\n")
    
    print("🚀 Starting training...")
    
    for episode in range(1, cfg.num_episodes + 1):
        obs, info = env.reset()
        done = truncated = False
        step = 0
        total_reward = 0
        
        # Per-agent storage
        ep_obs = [[] for _ in range(num_agents)]
        ep_acts = [[] for _ in range(num_agents)]
        ep_logp = [[] for _ in range(num_agents)]
        ep_rews = [[] for _ in range(num_agents)]
        ep_vals = [[] for _ in range(num_agents)]
        
        while not (done or truncated) and step < cfg.max_steps:
            actions = []
            for i in range(num_agents):
                # Get agent observation
                obs_i = obs[i].astype(np.float32)
                obs_t = torch.FloatTensor(obs_i).unsqueeze(0).to(device)
                
                # Forward pass
                dist = actors[i](obs_t)
                val = critics[i](obs_t)
                act = dist.sample()
                logp = dist.log_prob(act)
                
                actions.append(int(act.item()))
                ep_obs[i].append(obs_i)
                ep_acts[i].append(act.item())
                ep_logp[i].append(logp.squeeze())
                ep_vals[i].append(val.squeeze())
            
            # Environment step
            next_obs, reward, term, trunc, info = env.step(np.array(actions))
            done = term or trunc
            total_reward += reward
            step += 1
            
            # Shared reward (cooperative MARL)
            for i in range(num_agents):
                ep_rews[i].append(reward)
            
            obs = next_obs
        
        # Update networks (simple A2C-style)
        for i in range(num_agents):
            # Compute returns
            rews = np.array(ep_rews[i])
            returns = []
            G = 0
            for r in reversed(rews):
                G = r + cfg.gamma * G
                returns.insert(0, G)
            
            returns_t = torch.FloatTensor(returns).to(device)
            obs_t = torch.FloatTensor(np.array(ep_obs[i])).to(device)
            
            # Critic loss
            vals_pred = critics[i](obs_t)
            loss_c = F.mse_loss(vals_pred, returns_t)
            critic_opts[i].zero_grad()
            loss_c.backward()
            critic_opts[i].step()
            
            # Actor loss (basic policy gradient)
            dist = actors[i](obs_t)
            logp_new = dist.log_prob(torch.LongTensor(ep_acts[i]).to(device))
            advantages = returns_t - vals_pred.detach()
            loss_a = -(logp_new * advantages).mean()
            actor_opts[i].zero_grad()
            loss_a.backward()
            actor_opts[i].step()
        
        # Log episode
        picked = info.get('items_picked', len(rews))
        with open(csv_path, "a") as f:
            f.write(f"{episode},{total_reward:.1f},{step},{picked}\n")
        
        if episode % 10 == 0:
            print(f"✅ Ep {episode}: R={total_reward:.1f}, steps={step}, picked={picked}")
    
    print(f"🎉 Training complete! Results: logs/iac.csv")

if __name__ == "__main__":
    main()
