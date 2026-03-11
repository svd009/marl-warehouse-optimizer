import sys
sys.path.insert(0, './src')
from env.warehouse_env import WarehouseEnv
from agents.manager_worker import ManagerAgent, WorkerPolicy

def run_baseline(num_episodes=10):
    env = WarehouseEnv()
    manager = ManagerAgent(grid_size=10, num_agents=4)
    workers = [WorkerPolicy() for _ in range(4)]  # adjust if needed
    
    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done and steps < 100:
            # Manager assigns zones
            manager.assign_zones(obs)  # adjust method call
            
            actions = []
            for i, worker in enumerate(workers):
                action = worker.act(obs[i], manager.zones[i])  # adjust API
                actions.append(action)
            
            obs, r, done, _, info = env.step(actions)
            total_r += r
            steps += 1
        
        print(f"Baseline Ep {ep}: R={total_r:.1f}, steps={steps}")
        results.append((total_r, steps))
    
    avg_r = sum(r for r,s in results)/len(results)
    print(f"Baseline AVG: R={avg_r:.1f}")

if __name__ == "__main__":
    run_baseline()
