import numpy as np
import pandas as pd

from . import ObjectBuilder


def draw_line(p1, p2, spacing, max_w, max_h):

    d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    nb_points = int(d / spacing)

    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    positions = np.zeros((nb_points + 1, 2))
    idxs = np.arange(0, nb_points)

    positions[:-1, 0] = p1[0] + idxs * x_spacing
    positions[:-1, 1] = p1[1] + idxs * y_spacing
    positions[-1] = p2

    # # Remove pixels outisde
    # idxs = np.where((positions[:, 0] < 0) + (positions[:, 1] < 0) + (positions[:, 0] >= max_w) + (positions[:, 1] >= max_h))[0]
    # positions = positions[~idxs]

    return positions


class SimpleMicrotubuleBuilder(ObjectBuilder):
    name = "simple_microtubule"

    def build(self, image, verbose=False):

        n_frames = self.model.acquisition.n_frames
        tpf = self.model.acquisition.tpf
        pixel_size = self.model.get_pixel_size()

        n_microtubules = int(np.round(self.distribution(self.object_model.parameters["n_microtubules"], size=1)[0]))

        width = self.model.microscope.camera.chip_size_width
        height = self.model.microscope.camera.chip_size_height

        positions = np.empty((0, 2))
        frames = np.empty(0)
        n_fluos = np.empty(0)

        data = []

        for n in range(n_microtubules):

            initial_length = self.distribution(self.object_model.parameters["initial_length"], size=1)[0]
            nucleation_rate = self.distribution(self.object_model.parameters["nucleation_rate"], size=1)[0]

            angle = np.deg2rad(np.random.choice(range(0, 360)))

            seed = {}

            seed["start_x"] = (np.random.rand(1) * width * pixel_size)[0]
            seed["start_y"] = (np.random.rand(1) * height * pixel_size)[0]

            seed["end_x"] = seed["start_x"] + initial_length * np.sin(angle)
            seed["end_y"] = seed["start_y"] + initial_length * np.cos(angle)

            p1 = [seed["start_x"], seed["start_y"]]
            p2 = [seed["end_x"], seed["end_y"]]
            seed_positions = draw_line(p1, p2, spacing=self.object_model.parameters["spacing"],
                                       max_w=width, max_h=height)

            seed_n_fluos = self.distribution(self.object_model.parameters["n_fluos"], size=1)
            for frame in range(n_frames):

                datum = {}
                datum["mt_id"] = n
                datum["type"] = "seed"
                datum["frame"] = frame

                datum["start_x"] = seed["start_x"]
                datum["start_y"] = seed["start_y"]
                datum["end_x"] = seed["end_x"]
                datum["end_y"] = seed["end_y"]

                data.append(datum)

                positions = np.vstack([positions, seed_positions])

                current_frames = np.repeat(frame, seed_positions.shape[0])
                frames = np.concatenate([frames, current_frames])

                current_n_fluos = np.repeat(seed_n_fluos, seed_positions.shape[0])
                n_fluos = np.concatenate([n_fluos, current_n_fluos])

            # Should it nucleate?
            if np.random.rand() < nucleation_rate:

                start_x = seed['end_x']
                start_y = seed['end_y']
                direction = 0

                growth_rate = self.distribution(self.object_model.parameters["growth_rate"], size=1)[0]
                growth_rate_frame = (growth_rate * tpf) / 60

                spot_n_fluos = self.distribution(self.object_model.parameters["n_fluos"], size=1)
                for frame in range(n_frames):
                    length = growth_rate_frame * frame

                    end_x = start_x + length * np.sin(angle + direction)
                    end_y = start_y + length * np.cos(angle + direction)

                    datum = {}
                    datum["mt_id"] = n
                    datum["type"] = "mt"
                    datum["frame"] = frame

                    datum["start_x"] = start_x
                    datum["start_y"] = start_y
                    datum["end_x"] = end_x
                    datum["end_y"] = end_y

                    data.append(datum)

                    p1 = [start_x, start_y]
                    p2 = [end_x, end_y]
                    current_positions = draw_line(p1, p2, spacing=self.object_model.parameters["spacing"],
                                                  max_w=width, max_h=height)
                    current_positions = current_positions[1:]

                    positions = np.vstack([positions, current_positions])

                    current_frames = np.repeat(frame, current_positions.shape[0])
                    frames = np.concatenate([frames, current_frames])

                    current_n_fluos = np.repeat(spot_n_fluos, current_positions.shape[0])
                    n_fluos = np.concatenate([n_fluos, current_n_fluos])

        data = pd.DataFrame(data)
        data[["start_x", "start_y", "end_x", "end_y"]] /= pixel_size
        self._objects = data.to_dict()

        return self.discretize_fluorophores_list(image, positions.T, n_fluos, frames)
