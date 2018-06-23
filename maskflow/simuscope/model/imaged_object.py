from .serializer import Serializer


class ImagedObject(Serializer):

    _serializable = ["channels", "parameters"]

    def __init__(self):
        self.name = None
        self.model_name = None
        self.channels = None
        self.parameters = None
        self.photons_per_fluorophores = None
