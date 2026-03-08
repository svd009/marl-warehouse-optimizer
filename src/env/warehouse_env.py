import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict

class WarehouseEnv(gym.Env):
    def __init__(self, size: int = 10, num_agents: int = 4):
        super().__init__()
        self.size = size
        self.num_agents = num_agents
        self.num_items = size * 3
        
        self.action_space = spaces.MultiDiscrete([6] * num_agents)
        obs_size = size * size * 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_agents, obs_size), dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_pos = np.random.randint(0, self.size, (self.num_agents, 2))
        
        item_locs = []
        while len(item_locs) < self.num_items:
            loc = np.random.randint(0, self.size, 2)
            if not any(np.array_equal(loc, pos) for pos in self.agent_pos):
                item_locs.append(loc)
        self.items = np.array(item_locs)
        
        self.picked_items = 0
        obs = self._get_obs()
        return obs, {}
    
    def step(self, actions):
        reward = 0
        done = False
        
        for agent_id, action in enumerate(actions):
            pos = self.agent_pos[agent_id].copy()
            
            if action < 4:
                if action == 0: pos[0] -= 1
                elif action == 1: pos[0] += 1
                elif action == 2: pos[1] -= 1
                elif action == 3: pos[1] += 1
                pos = np.clip(pos, 0, self.size - 1)
                self.agent_pos[agent_id] = pos
            
            elif action == 5:  # Pick nearest item
                if len(self.items) == 0:
                    continue
                distances = np.linalg.norm(self.items - pos, axis=1)
                nearest_item = np.argmin(distances)
                if distances[nearest_item] <= 1.5:
                    self.items = np.delete(self.items, nearest_item, axis=0)
                    self.picked_items += 1
                    reward += 10
        
        reward += self.picked_items * 0.1
        if len(self.items) == 0: 
            done = True
            reward += 100
        
        obs = self._get_obs()
        return obs, reward, done, False, {}
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.num_agents, self.size * self.size * 2))
        for i, pos in enumerate(self.agent_pos):
            grid_flat = np.zeros(self.size * self.size * 2)
            agent_idx = pos[0] * self.size + pos[1]
            grid_flat[agent_idx] = 1
            
            item_grid = np.zeros(self.size * self.size)
            for item in self.items:
                idx = item[0] * self.size + item[1]
                item_grid[idx] = 1
            grid_flat[self.size*self.size:] = item_grid
            
            obs[i] = grid_flat
        return obs
