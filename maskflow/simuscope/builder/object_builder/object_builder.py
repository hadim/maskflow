import logging

import numpy as np

from ..abstract_builder import AbstractBuilder


class ObjectBuilder(AbstractBuilder):
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    @staticmethod
    def find(name):
        for plugin in ObjectBuilder.subclasses:
            if plugin.name == name:
                return plugin
        return None

    def __init__(self, object_model, model, object_name):
        super().__init__(model)
        self.object_model = object_model
        self.object_name = object_name
        self._objects = None

    def build(self, image, verbose=False):
        super().build(image, verbose)

    def get_channel_indexes(self):
        channel_indexes = []
        for channel_name in self.object_model.channels:
            if channel_name in list(self.model.acquisition.channels.keys()):
                channel_indexes.append(list(self.model.acquisition.channels.keys()).index(channel_name))
        return channel_indexes

    def discretize_fluorophores_list(self, image, positions, n_fluos, frames):
        positions_pixel = np.round(positions / self.model.get_pixel_size(), 0).astype("int")
        channel_indexes = self.get_channel_indexes()

        n_photons = self.object_model.photons_per_fluorophores

        # Iterate over all the fluorophores
        for i in range(positions.shape[1]):
            x, y = positions_pixel[:, i]

            if x < 0 or y < 0:
                continue

            n_fluo = n_fluos[i]
            frame = frames[i]
            if not isinstance(frames[i], list):
                frame = int(frames[i])

            try:
                if isinstance(frame, list) or frame >= 0:
                    image[frame, channel_indexes, y, x] += n_fluo * n_photons
                else:
                    image[:, channel_indexes, y, x] += n_fluo * n_photons
            except IndexError:
                logging.debug("The specified fluorophores positions and/or frames are out the field of view.")

        return image

    def get_objects(self):
        return self._objects
