import numpy as np
from env.warehouse_env import WarehouseEnv
from agents.manager_worker import ManagerAgent
from agents.qlearning import QLearningWorker

def train_qlearning(episodes: int = 1000):
    """
    WHAT: Train 4 independent Q-learning workers + 1 heuristic manager
    WHY: Workers learn optimal paths within zones, manager handles coordination
    """
    env = WarehouseEnv(size=10, num_agents=4)
    manager = ManagerAgent(grid_size=10, num_zones=2)
    workers = [QLearningWorker(grid_size=10) for _ in range(4)]
    
    episode_rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        
        while True:
            # Manager assigns zones
            item_locations = env.items.copy()
            zones = manager.assign_zones(item_locations, 4)
            
            # Workers act
            actions = []
            for i, worker in enumerate(workers):
                action = worker.act(env.agent_pos[i], zones[i].id)
                actions.append(action)
            
            # Environment step + rewards
            obs, reward, done, _, _ = env.step(actions)
            total_reward += reward
            
            # Workers learn (simplified - use zone-based states)
            for i, worker in enumerate(workers):
                state = (int(env.agent_pos[i][0]), int(env.agent_pos[i][1]), zones[i].id)
                next_state = state  # Simplified for now
                worker.update(state, actions[i], reward/4, next_state)  # Share reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        if ep % 100 == 0:
            print(f"Episode {ep}: reward={total_reward:.1f}, items_picked={env.picked_items}")
    
    return episode_rewards

if __name__ == "__main__":
    rewards = train_qlearning(500)
    print(f"Final 100 episodes avg: {np.mean(rewards[-100:]):.1f}")
    print("Training complete! Q-tables learned.")
