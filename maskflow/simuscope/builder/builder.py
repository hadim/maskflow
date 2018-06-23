import logging
import json

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tifffile import imsave

from .image_builders import CameraBuilder
from .image_builders import ConvolutionBuilder
from .image_builders import BackgroundBuilder
from .object_builder import ObjectBuilder
from ..utils import get_memory_array
from ..model.imaged_object import ImagedObject


class Builder():

    def __init__(self, model):
        self.model = model
        self._init_image()
        self._init_builders()

    def _init_builders(self):
        self.objects = self._init_objects()
        self.background = BackgroundBuilder(self.model)
        self.convolution = ConvolutionBuilder(self.model)
        self.camera = CameraBuilder(self.model)

        self.builders = []
        self.builders.extend(self.objects)
        self.builders.append(self.background)
        self.builders.append(self.convolution)
        self.builders.append(self.camera)

    def _init_image(self):

        self.channel_names = list(self.model.acquisition.channels.keys())
        self.channel_colors = [self.model.acquisition.channels[key].color for key in self.channel_names]

        self.n_frames = self.model.acquisition.n_frames
        self.n_channels = len(self.model.acquisition.channels)
        self.pixel_width = self.model.microscope.camera.chip_size_width
        self.pixel_height = self.model.microscope.camera.chip_size_height
        shape = (self.n_frames, self.n_channels, self.pixel_width, self.pixel_height)

        self.original_image = np.zeros(shape=shape)
        self.image = self.original_image

    def _init_objects(self):
        object_builders = []
        for object_name, object_model in self.model.objects.items():
            object_model_name = object_model.model_name
            candidate = ObjectBuilder.find(object_model_name)
            if candidate:
                logging.info(f"Building with {object_model} builder")
                object_builders.append(candidate(object_model, self.model, object_name))
            else:
                logging.error(f"No candidates found for {object_model_name}")
        return object_builders

    def reset(self):
        self._init_image()

    def build(self, keep_images=False):

        self.reset()

        images = [self.image]
        for builder in self.builders:
            logging.info(f"Build : {builder}")
            self.image = builder.build(self.image)
            images.append(self.image)

        if keep_images:
            return images

    def build_without_objects(self):
        self.image = self.convolution.build(self.image)
        self.image = self.camera.build(self.image)

    def add_builder(self, name, builder_class, channels, photons_per_fluorophores, parameters={}):
        self.model.objects[name] = ImagedObject()
        self.model.objects[name].name = name
        self.model.objects[name].model_name = builder_class.name
        self.model.objects[name].channels = channels
        self.model.objects[name].photons_per_fluorophores = photons_per_fluorophores
        self.model.objects[name].parameters = parameters

        self._init_builders()

    def reset_objects(self):
        self.objects = []
        self.model.objects = {}
        self._init_builders()

    def __str__(self):
        s = ""
        s += f"Image shape: {self.image.shape}\n"
        s += f"Image memory size: {get_memory_array(self.image)}\n"
        s += f"Channels: {list(self.model.acquisition.channels.keys())}\n"
        s += f"Objects: {list(self.objects)}\n"
        return s

    def __repr__(self):
        return self.__str__()

    def show(self, image=None, frame=0, colormap="viridis", vmin=None, vmax=None, return_fig=False):
        """Show a single frame of a two-channels stack assuming order TCXY."""
        fig, axs = plt.subplots(1, self.n_channels, figsize=(6 * self.n_channels, 6),
                                sharex=True, sharey=True)

        if image is None:
            image = self.image

        # Display images
        for i, channel_name in enumerate(self.channel_names):

            if self.n_channels == 1:
                ax = axs
            else:
                ax = axs[i]

            img_ax = ax.imshow(image[frame, i, :, :], vmin=vmin, vmax=vmax,
                                   interpolation="none", aspect="equal", cmap=colormap)
            ax.set_title(channel_name)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(img_ax, cax=cax, orientation='vertical')

        fig.suptitle("Frame : {}".format(frame))
        if return_fig:
            return fig

    def show_interactive(self, image=None, colormap="viridis", vmin=None, vmax=None):
        import ipywidgets as widgets
        from ipywidgets import interact

        def display_fn(frame):
            self.show(image, frame, colormap, vmin, vmax, return_fig=True)

        slider = widgets.IntSlider(min=0, max=self.model.acquisition.n_frames - 1, step=1, value=0)
        interact(display_fn, frame=slider)

    def save_image(self, image_path):
        if self.model.microscope.camera.bitdepth <= 8:
            im = self.image.astype('uint8')
        elif self.model.microscope.camera.bitdepth <= 16:
            im = self.image.astype('uint16')
        elif self.model.microscope.camera.bitdepth <= 32:
            im = self.image.astype('uint32')
        else:
            logging.error(f"Incorrect bitdepth: '{self.model.microscope.camera.bitdepth}'")

        # Save images with ImageJ metadata for easy ImageJ opening.
        total_images = self.n_frames * self.n_channels
        ij_description = f"'ImageJ=1.51n\nimages={total_images}\nchannels={self.n_channels}\n"
        f"frames={self.n_frames}\nhyperstack=true\nmode=composite\nfinterval=1\nloop=false\n'"
        imsave(image_path, im, imagej=ij_description)

    def get_objects_as_dict(self):
        objs = {}
        for obj in self.objects:
            objs[obj.object_name] = obj.get_objects()
        return objs

    def save_objects(self, objects_path):
        with open(objects_path, "w") as f:
            f.write(json.dumps(self.get_objects_as_dict(), indent=2, default=int))

    def save(self, basepath):
        self.save_image(basepath + ".tif")
        self.save_objects(basepath + ".json")
