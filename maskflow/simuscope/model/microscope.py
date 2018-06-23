from .serializer import Serializer


class Microscope(Serializer):

    _serializable = ["psf_kernel_size", "objective", "camera"]

    def __init__(self):
        self.psf_kernel_size = None
        self.objective = {"magnification": None}
        self.camera = Camera()


class Camera(Serializer):

    _serializable = ["chip_size_width", "chip_size_height",
                     "pixel_area_width", "pixel_area_height",
                     "qe", "dark_noise", "bitdepth",
                     "sensitivity", "baseline"]

    def __init__(self):
        self.chip_size_width = None
        self.chip_size_height = None

        self.pixel_area_width = None
        self.pixel_area_height = None

        self.qe = None
        self.dark_noise = None
        self.bitdepth = None
        self.sensitivity = None
        self.baseline = None
