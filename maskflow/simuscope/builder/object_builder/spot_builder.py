import numpy as np
import pandas as pd

from . import ObjectBuilder


class SpotBuilder(ObjectBuilder):
    name = "spot"

    def build(self, image, verbose=False):
        positions = np.array([self.object_model.parameters["x"], self.object_model.parameters["y"]])
        n_fluos = np.array(self.object_model.parameters["n_fluorophores"])
        frames = self.object_model.parameters["frame"]

        df = pd.DataFrame([positions[0], positions[1], n_fluos, frames]).T
        df.columns = ["x", "y", "n_fluos", "frames"]
        self._objects = df.to_dict()

        return self.discretize_fluorophores_list(image, positions, n_fluos, frames)


class RandomSpotBuilder(ObjectBuilder):
    name = "random_spot"

    def build(self, image, verbose=False):

        n_spots = int(np.round(self.distribution(self.object_model.parameters["n_spots"], size=1)[0]))
        n_fluos = self.distribution(self.object_model.parameters["n_fluos"], size=n_spots)
        frames = np.random.randint(0, self.model.acquisition.n_frames, size=n_spots)

        pixel_size = self.model.get_pixel_size()
        x = np.random.rand(n_spots) * self.model.microscope.camera.chip_size_width * pixel_size
        y = np.random.rand(n_spots) * self.model.microscope.camera.chip_size_height * pixel_size
        positions = np.array([x, y])

        df = pd.DataFrame([x / pixel_size, y / pixel_size, n_fluos, frames]).T
        df.columns = ["x", "y", "n_fluos", "frames"]
        self._objects = df.to_dict()

        return self.discretize_fluorophores_list(image, positions, n_fluos, frames)
