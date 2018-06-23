import os
import tempfile

from simuscope import Model


def test_empty_model():
    model = Model()
    assert not model.microscope.psf_kernel_size


def test_simple_model():
    model = Model.load_model("simple_model")
    assert model.microscope.psf_kernel_size == 15


def test_no_objects_model():
    model = Model.load_model("no_objects")
    assert len(model.objects) == 0


def test_brownian_motion_model():
    model = Model.load_model("brownian_motion")
    assert model.objects["spot1"].parameters["d_coeff"] == 0.01


def test_save_load():
    model = Model()
    model.microscope.psf_kernel_size = 666
    _, tmp_path = tempfile.mkstemp()
    model.to_yaml(tmp_path)

    model = Model.load_from_yaml(tmp_path)
    assert model.microscope.psf_kernel_size == 666

    os.remove(tmp_path)
