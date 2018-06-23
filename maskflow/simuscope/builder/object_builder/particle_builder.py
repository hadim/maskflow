import numpy as np
import pandas as pd

from . import ObjectBuilder


class BrownianParticleBuilder(ObjectBuilder):
    name = "brownian_particle"

    def build(self, image, verbose=False):

        n_frames = self.model.acquisition.n_frames
        tpf = self.model.acquisition.tpf

        dt = self.object_model.parameters["dt"]
        d_coeff = self.object_model.parameters["d_coeff"]
        n_spots = int(np.round(self.distribution(self.object_model.parameters["n_spots"], size=1)[0]))

        k = np.sqrt(2 * d_coeff * dt)
        total_duration = n_frames * tpf  # s

        time = np.arange(0, total_duration, dt)
        n_step = time.shape[0]

        n_dimension = 2

        positions = np.empty((0, 2))
        frames = np.empty(0)
        n_fluos = np.empty(0)

        data = pd.DataFrame([])

        for n in range(n_spots):

            datum = {}
            datum["spot_id"] = n

            # Get random displacement
            dp = k * np.random.randn(n_step, n_dimension)

            # Setup custom initial position
            pixel_size = self.model.get_pixel_size()
            initial_position_x = np.random.rand(1) * self.model.microscope.camera.chip_size_width * pixel_size
            initial_position_y = np.random.rand(1) * self.model.microscope.camera.chip_size_height * pixel_size
            dp[0] = np.array([initial_position_x, initial_position_y]).T

            # Get position
            p = np.cumsum(dp, axis=0)

            # Now we only keep the positions that correspond to the acquired frames
            acquired_time = np.arange(0, n_frames) * tpf
            indexes_to_keep = [np.argmin(np.abs(time - t)) for t in acquired_time]

            current_positions = p[indexes_to_keep]
            positions = np.vstack([positions, current_positions])

            current_frames = np.arange(0, n_frames)
            frames = np.concatenate([frames, current_frames])

            spot_n_fluos = self.distribution(self.object_model.parameters["n_fluos"], size=1)
            current_n_fluos = np.repeat(spot_n_fluos, current_positions.shape[0])
            n_fluos = np.concatenate([n_fluos, current_n_fluos])

            df = pd.DataFrame(p[indexes_to_keep] / pixel_size, columns=["x", "y"])
            df["spot_id"] = n
            df["n_fluos"] = current_n_fluos
            df["frames"] = current_frames
            data = data.append(df)

        self._objects = data.to_dict()

        return self.discretize_fluorophores_list(image, positions.T, n_fluos, frames)
