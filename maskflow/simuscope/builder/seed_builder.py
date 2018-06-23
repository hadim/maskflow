import numpy as np


class SeedBuilder():
    def __init__(self, model):
        self.model = model
        self.max_w = self.model.imaging.fov["width"]
        self.max_h = self.model.imaging.fov["height"]
        self.seeds = []

    def create_seeds(self, n):
        for i in range(n):
            self.create_seed()
        return self.seeds

    def create_seed(self):
        seed = {}
        seed["id"] = len(self.seeds) + 1
        seed["length"] = self.model.seeds.length.value()
        seed["angle"] = np.deg2rad(np.random.choice(range(0, 360)))

        seed["start_x"] = np.random.choice(range(0, self.max_w))
        seed["start_y"] = np.random.choice(range(0, self.max_h))
        seed["end_x"] = int(np.round(seed["start_x"] + seed["length"] * np.sin(seed["angle"]), 0))
        seed["end_y"] = int(np.round(seed["start_y"] + seed["length"] * np.cos(seed["angle"]), 0))

        self.seeds.append(seed)
        return seed
