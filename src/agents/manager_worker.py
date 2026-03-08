from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Zone:
    id: int
    row_start: int
    row_end: int

class ManagerAgent:
    def __init__(self, grid_size: int, num_zones: int):
        self.grid_size = grid_size
        self.num_zones = num_zones
        self.zones = self._create_zones()

    def _create_zones(self) -> List[Zone]:
        rows_per_zone = self.grid_size // self.num_zones
        zones = []
        for i in range(self.num_zones):
            start = i * rows_per_zone
            end = (i + 1) * rows_per_zone if i < self.num_zones - 1 else self.grid_size
            zones.append(Zone(id=i, row_start=start, row_end=end))
        return zones

    def assign_zones(self, item_locations: np.ndarray, num_workers: int) -> List[Zone]:
        counts = np.zeros(self.num_zones, dtype=int)
        for item in item_locations:
            row = item[0]
            for z in self.zones:
                if z.row_start <= row < z.row_end:
                    counts[z.id] += 1
                    break
        zone_order = np.argsort(-counts)
        return [self.zones[zone_order[i % self.num_zones]] for i in range(num_workers)]

class WorkerPolicy:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def act(self, agent_pos: np.ndarray, item_locations: np.ndarray, zone: Zone) -> int:
        zone_items = [item for item in item_locations if zone.row_start <= item[0] < zone.row_end]
        if not zone_items:
            return 4  # stay
        
        zone_items = np.array(zone_items)
        distances = np.linalg.norm(zone_items - agent_pos, axis=1)
        nearest_item = zone_items[np.argmin(distances)]

        if np.linalg.norm(nearest_item - agent_pos) <= 1.1:
            return 5  # pick
        
        dr = nearest_item[0] - agent_pos[0]
        dc = nearest_item[1] - agent_pos[1]
        if abs(dr) > abs(dc):
            return 1 if dr > 0 else 0  # down or up
        return 3 if dc > 0 else 2  # right or left
