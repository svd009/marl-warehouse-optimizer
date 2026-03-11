import os
import csv
import sys
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.warehouse_env import WarehouseEnv
from src.agents.manager_worker import ManagerAgent
from src.agents.networks import WorkerActorNet, WorkerCriticNet

@dataclass
class TrainConfig:
    num_episodes: int = 50  # Reduced for quick test
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    rollout_len: int = 200
    log_dir: str = "logs"
    log_csv: str = "logs/iac_train.csv"
    device: str = "cpu"

def make_env():
    return WarehouseEnv()

def prepare_logging(cfg):
    os.makedirs(cfg.log_dir, exist_ok=True)
    if not os.path.exists(cfg.log_csv):
        with open(cfg.log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode","total_reward","items_picked","items_remaining","steps","pick_rate","completion_flag"])

def log_episode(cfg, episode, total_reward, items_picked, items_remaining, steps):
    pick_rate = items_picked / max(steps, 1)
    completion_flag = 1 if items_remaining == 0 else 0
    with open(cfg.log_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward, items_picked, items_remaining, steps, pick_rate, completion_flag])

def obs_for_agent(obs, agent_idx):
    return obs[agent_idx]

def train():
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    print("Creating environment...")
    env = make_env()
    
    # Hardcoded for your 10x10 grid, 4 agents, 6 actions
    num_agents = 4
    grid_size = 10
    single_obs_dim = grid_size * grid_size * 2  # 200
    n_actions = 6

    print(f"Obs dim: {single_obs_dim}, Actions: {n_actions}")
    
    # Skip manager for now (commented out)
    # manager = ManagerAgent(grid_size=grid_size, num_agents=num_agents)

    print("Creating networks...")
    actors = [WorkerActorNet(single_obs_dim, n_actions).to(device) for _ in range(num_agents)]
    critics = [WorkerCriticNet(single_obs_dim).to(device) for _ in range(num_agents)]
    actor_optimizers = [optim.Adam(actor.parameters(), lr=cfg.actor_lr) for actor in actors]
    critic_optimizers = [optim.Adam(critic.parameters(), lr=cfg.critic_lr) for critic in critics]

    prepare_logging(cfg)
    print("Starting training...")

    for episode in range(1, cfg.num_episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        total_reward = 0.0

        ep_obs = [[] for _ in range(num_agents)]
        ep_actions = [[] for _ in range(num_agents)]
        ep_log_probs = [[] for _ in range(num_agents)]
        ep_rewards = [[] for _ in range(num_agents)]
        ep_values = [[] for _ in range(num_agents)]
        ep_dones = []

        while not (done or truncated) and step < cfg.rollout_len:
            actions = []
            for i in range(num_agents):
                agent_obs_np = obs_for_agent(obs, i).astype(np.float32)
                agent_obs_t = torch.from_numpy(agent_obs_np).unsqueeze(0).to(device)

                # Actor
                dist = actors[i](agent_obs_t)
                value = critics[i](agent_obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                actions.append(int(action.item()))
                ep_obs[i].append(agent_obs_np)
                ep_actions[i].append(action.item())
                ep_log_probs[i].append(log_prob.squeeze(0))
                ep_values[i].append(value.squeeze(0))

            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            done = terminated or truncated
            total_reward += reward
            step += 1
            ep_dones.append(done)

            # Shared reward for all agents (cooperative)
            for i in range(num_agents):
                ep_rewards[i].append(reward)

            obs = next_obs

        # Simple backward pass (critic only for now)
        for i in range(num_agents):
            rewards = np.array(ep_rewards[i])
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + cfg.gamma * G
                returns.insert(0, G)
            
            returns_t = torch.FloatTensor(returns).to(device)
            obs_batch = torch.FloatTensor(np.array(ep_obs[i])).to(device)
            values_pred = critics[i](obs_batch)
            
            critic_loss = torch.nn.functional.mse_loss(values_pred.squeeze(), returns_t)
            critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critics[i].parameters(), 0.5)
            critic_optimizers[i].step()

        # Log results
        items_picked = getattr(info, 'items_picked', 0) if hasattr(info, 'items_picked') else 0
        items_remaining = getattr(info, 'items_remaining', 30) if hasattr(info, 'items_remaining') else 30
        log_episode(cfg, episode, total_reward, items_picked, items_remaining, step)

        if episode % 10 == 0:
            print(f"Episode {episode}: R={total_reward:.1f}, steps={step}, picked={items_picked}, remaining={items_remaining}")

    print(f"Training complete! Check logs/iac_train.csv")

if __name__ == "__main__":
    train()
