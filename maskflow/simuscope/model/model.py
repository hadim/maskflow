import pathlib

from .serializer import Serializer
from .microscope import Microscope
from .acquisition import Acquisition
from .imaged_object import ImagedObject

from ..builder import Builder

DATA_DIR = pathlib.Path(__file__).parents[1] / "data"


class Model(Serializer):
    _serializable = ["version", "microscope", "acquisition", ("objects", ImagedObject)]

    def __init__(self):
        self.version = 1
        self.microscope = Microscope()
        self.acquisition = Acquisition()
        self.objects = {}

        self._validate()

    @staticmethod
    def load_from_yaml(yaml_path):
        model = Model()
        model.from_yaml(yaml_path)
        model._validate()
        return model

    @staticmethod
    def load_model(model_name):
        return Model.load_from_yaml(DATA_DIR / (model_name + ".yaml"))

    def get_builder(self):
        builder = Builder(self)
        return builder

    def get_pixel_size(self):
        pixel_size = self.microscope.camera.pixel_area_width / self.microscope.objective["magnification"]
        return pixel_size
