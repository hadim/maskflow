import numpy as np
from scipy import stats
from scipy import ndimage

from .abstract_builder import AbstractBuilder
from .psf import PSFBuilder


class CameraBuilder(AbstractBuilder):
    """From http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations.html."""

    def build(self, image, verbose=False):
        # The input self.image needs to contain a number of photons when this method is called.

        # Add shot noise
        photons = stats.poisson.random_state.poisson(image, size=image.shape)

        # Convert to electrons
        electrons = self.model.microscope.camera.qe * photons

        # Add dark noise
        electrons_out = stats.norm.random_state.normal(scale=self.model.microscope.camera.dark_noise,
                                                       size=electrons.shape)
        electrons_out += electrons

        # Convert to ADU and add baseline
        max_adu = np.int(2**self.model.microscope.camera.bitdepth - 1)

        # Convert to discrete numbers
        adu = (electrons_out * self.model.microscope.camera.sensitivity).astype(np.int)

        # Models pixel saturation
        adu[adu > max_adu] = max_adu

        # Add baseline values
        adu += self.model.microscope.camera.baseline

        return adu


class ConvolutionBuilder(AbstractBuilder):

    def _init_builder(self):
        # Look for all PSFBuilder subclass that match with the PSF model name
        # defined in the model object.
        candidates = {}
        self.channel_names = list(self.model.acquisition.channels.keys())
        for channel in self.channel_names:
            candidates[channel] = []

        for psf_plugin in PSFBuilder.subclasses:
            for channel in self.channel_names:
                if psf_plugin.name == self.model.acquisition.channels[channel].psf.model_name:
                    candidates[channel].append(psf_plugin)

        # Select the first candidates or raise an error if no candidates
        self.psf_builders = {}
        for channel, psf_builder in candidates.items():
            if len(psf_builder) == 0:
                raise Exception("No PSF model available for channel '{channel}' and model"
                                " name '{self.model.acquisition.channels[channel].psf.model_name}'")
            self.psf_builders[channel] = psf_builder[0]

    def build(self, image, verbose=False):

        kernel_size_pixel = self.model.microscope.psf_kernel_size
        new_im = np.zeros(shape=image.shape)
        for i, channel in enumerate(self.channel_names):
            sigma_pixel = self.model.acquisition.channels[channel].psf.parameters["sigma"]
            sigma_pixel /= self.model.get_pixel_size()
            psf = self.psf_builders[channel]().generate(kernel_size_pixel, sigma_pixel)
            for frame in range(self.model.acquisition.n_frames):
                new_im[frame, i] = ndimage.filters.convolve(image[frame, i], psf)
        return new_im


class BackgroundBuilder(AbstractBuilder):

    def build(self, image, verbose=False):

        for i, (channel_name, channel) in enumerate(self.model.acquisition.channels.items()):
            if channel.snr:

                single_channel_image = image[:, i]
                if single_channel_image.sum() > 0:
                    signal_indexes = np.where(single_channel_image > 0)
                    background_indexes = np.where(~(single_channel_image > 0))

                    signal_mean = single_channel_image[signal_indexes].mean()

                    background_value = signal_mean / channel.snr
                    single_channel_image[background_indexes] = background_value

        return image