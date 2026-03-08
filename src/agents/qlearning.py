import numpy as np
from collections import defaultdict
from typing import Dict, Tuple

class QLearningWorker:
    def __init__(self, grid_size: int, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.grid_size = grid_size
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def _state_to_tuple(self, agent_pos: np.ndarray, zone_id: int) -> Tuple:
        return (int(agent_pos[0]), int(agent_pos[1]), zone_id)
    
    def act(self, agent_pos: np.ndarray, zone_id: int) -> int:
        state = self._state_to_tuple(agent_pos, zone_id)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 6)
        
        q_values = self.q_table[state]
        if not q_values:
            return np.random.randint(0, 6)
        
        return max(q_values, key=q_values.get)
    
    def update(self, state, action: int, reward: float, next_state):
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
