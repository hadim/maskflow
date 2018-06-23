import numpy as np
from scipy import signal


class PSFBuilder():
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    def generate(self, *args, **kwargs):
        pass


class GaussianPSF(PSFBuilder):
    name = "gaussian"

    def generate(self, kernel_size_pixel, sigma_pixel):
        gaussian_kernel_1d = signal.gaussian(kernel_size_pixel, std=sigma_pixel)
        gaussian_kernel_1d = gaussian_kernel_1d.reshape(kernel_size_pixel, 1)
        gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
        return gaussian_kernel_2d
