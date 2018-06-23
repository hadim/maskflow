from .serializer import Serializer


class PSF(Serializer):

    _serializable = ["model_name", "parameters"]

    def __init__(self):
        self.model_name = None
        self.parameters = {}


class Channel(Serializer):

    _serializable = ["color", "snr", "psf"]

    def __init__(self):
        self.color = None
        self.snr = None
        self.psf = PSF()


class Acquisition(Serializer):

    _serializable = ["tpf", "n_frames", ("channels", Channel)]

    def __init__(self):
        self.tpf = None
        self.n_frames = None
        self.channels = {}
