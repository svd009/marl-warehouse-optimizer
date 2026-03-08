import numpy as np
from env.warehouse_env import WarehouseEnv
from agents.manager_worker import ManagerAgent, WorkerPolicy


def run_episode(
    grid_size: int = 10,
    num_agents: int = 4,
    num_zones: int = 2,
    max_steps: int = 200,
):
    env = WarehouseEnv(size=grid_size, num_agents=num_agents)
    manager = ManagerAgent(grid_size=grid_size, num_zones=num_zones)
    workers = [WorkerPolicy(grid_size=grid_size) for _ in range(num_agents)]

    obs, info = env.reset()
    total_reward = 0
    step = 0

    while True:
        item_locations = env.items.copy()
        zone_assignments = manager.assign_zones(item_locations, num_agents)

        actions = []
        for agent_id in range(num_agents):
            agent_pos = env.agent_pos[agent_id]
            zone = zone_assignments[agent_id]
            action = workers[agent_id].act(agent_pos, item_locations, zone)
            actions.append(action)

        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward
        step += 1

        if done or truncated or step >= max_steps:
            break

    return {
        "total_reward": total_reward,
        "steps": step,
        "items_remaining": len(env.items),
        "items_picked": env.picked_items,
    }

if __name__ == "__main__":
    stats = run_episode()
    print("Episode finished.")
    print(f"Total reward: {stats['total_reward']}")
    print(f"Steps: {stats['steps']}")
    print(f"Items picked: {stats['items_picked']}")
    print(f"Items remaining: {stats['items_remaining']}")
