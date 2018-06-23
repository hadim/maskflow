from simuscope import Model
from simuscope import ObjectBuilder


def test_builder_simple_model():
    model = Model.load_model("simple_model")
    model.microscope.camera.chip_size_height = 200
    model.microscope.camera.chip_size_width = 200
    model.acquisition.n_frames = 10

    builder = model.get_builder()
    assert builder.image.shape == (10, 2, 200, 200)

    builder.build()
    objs = builder.get_objects_as_dict()
    assert isinstance(objs, dict)


def test_builder_no_objects():
    model = Model.load_model("no_objects")
    model.microscope.camera.chip_size_height = 200
    model.microscope.camera.chip_size_width = 200
    model.acquisition.n_frames = 10

    builder = model.get_builder()
    assert builder.image.shape == (10, 2, 200, 200)

    builder.build()
    objs = builder.get_objects_as_dict()
    assert isinstance(objs, dict)


def test_builder_brownian_motion():
    model = Model.load_model("brownian_motion")
    model.microscope.camera.chip_size_height = 20
    model.microscope.camera.chip_size_width = 20
    model.acquisition.n_frames = 10

    builder = model.get_builder()
    assert builder.image.shape == (10, 2, 20, 20)

    builder.build()
    objs = builder.get_objects_as_dict()
    assert isinstance(objs, dict)


def test_custom_object():
    class SquareBuilder(ObjectBuilder):
        name = "square_object"

        def build(self, image, verbose=False):
            channel_indexes = self.get_channel_indexes()
            image[:, channel_indexes, 105:155, 105:155] += 100
            return image

    model = Model.load_model("no_objects")
    model.microscope.camera.chip_size_height = 200
    model.microscope.camera.chip_size_width = 200

    builder = model.get_builder()

    builder.reset_objects()
    builder.add_builder("square_object", SquareBuilder, ["channel_2"], 200, parameters={})
    builder.build()


def test_builder_simple_microtubule():
    model = Model.load_model("simple_microtubule")
    model.microscope.camera.chip_size_height = 200
    model.microscope.camera.chip_size_width = 200
    model.acquisition.n_frames = 10

    builder = model.get_builder()
    assert builder.image.shape == (10, 2, 200, 200)

    builder.build()
    objs = builder.get_objects_as_dict()
    assert isinstance(objs, dict)
