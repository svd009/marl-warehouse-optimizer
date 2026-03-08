import torch
import numpy as np
from env.warehouse_env import WarehouseEnv
from agents.manager_worker import ManagerAgent
from agents.pg_policy import PolicyNetwork

def train_policy_gradient(episodes=200):
    env = WarehouseEnv(10, 4)
    manager = ManagerAgent(10, 2)
    
    # 4 workers, each with own policy
    policies = [PolicyNetwork(obs_size=200) for _ in range(4)]
    optimizers = [torch.optim.Adam(p.parameters(), lr=0.01) for p in policies]
    
    episode_rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs, rewards = [], []
        
        while True:
            # Manager assigns zones
            zones = manager.assign_zones(env.items.copy(), 4)
            
            # Workers act (policy gradient)
            actions = []
            for i in range(4):
                obs_tensor = torch.FloatTensor(obs[i])
                probs = policies[i](obs_tensor)
                action = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[action])
                
                log_probs.append(log_prob)
                actions.append(action)
            
            # Step environment
            obs, reward, done, _, _ = env.step(actions)
            rewards.append(reward)
            
            if done: break
        
        # Policy gradient update (REINFORCE)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.95 * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        for i, optimizer in enumerate(optimizers):
            optimizer.zero_grad()
            loss = torch.stack(policy_loss[i::4]).sum()  # Each policy's own loss
            loss.backward()
            optimizer.step()
        
        episode_rewards.append(sum(rewards))
        
        if ep % 50 == 0:
            print(f"Ep {ep}: reward={sum(rewards):.1f}, items={env.picked_items}")
    
    print(f"Final avg: {np.mean(episode_rewards[-50:]):.1f}")
