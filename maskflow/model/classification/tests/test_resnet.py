import numpy.testing as npt

import maskflow


def test_resnet_18():
    num_classes = 10
    input_shape = (20, 224, 224, 3)

    model = maskflow.model.ResNet18(num_classes=num_classes, include_top=True, data_format='channels_last')
    model.build(input_shape=input_shape)

    print(model.count_params())
    assert model.count_params() == 11194826

    output_shape = model.compute_output_shape(input_shape)
    print(output_shape)
    npt.assert_equal(output_shape, [20, 10])


def test_resnet_34():
    num_classes = 10
    input_shape = (20, 224, 224, 3)

    model = maskflow.model.ResNet34(num_classes=num_classes, include_top=True, data_format='channels_last')
    model.build(input_shape=input_shape)

    print(model.count_params())
    assert model.count_params() == 21314122

    output_shape = model.compute_output_shape(input_shape)
    print(output_shape)
    npt.assert_equal(output_shape, [20, 10])


def test_resnet_50():
    num_classes = 10
    input_shape = (20, 224, 224, 3)

    model = maskflow.model.ResNet50(num_classes=num_classes, include_top=True, data_format='channels_last')
    model.build(input_shape=input_shape)

    print(model.count_params())
    assert model.count_params() == 23584906

    output_shape = model.compute_output_shape(input_shape)
    print(output_shape)
    npt.assert_equal(output_shape, [20, 10])


def test_resnet_101():
    num_classes = 10
    input_shape = (20, 224, 224, 3)

    model = maskflow.model.ResNet101(num_classes=num_classes, include_top=True, data_format='channels_last')
    model.build(input_shape=input_shape)

    print(model.count_params())
    assert model.count_params() == 42655370

    output_shape = model.compute_output_shape(input_shape)
    print(output_shape)
    npt.assert_equal(output_shape, [20, 10])


def test_resnet_152():
    num_classes = 10
    input_shape = (20, 224, 224, 3)

    model = maskflow.model.ResNet152(num_classes=num_classes, include_top=True, data_format='channels_last')
    model.build(input_shape=input_shape)

    print(model.count_params())
    assert model.count_params() == 58368138

    output_shape = model.compute_output_shape(input_shape)
    print(output_shape)
    npt.assert_equal(output_shape, [20, 10])


# Disable bceause it takes time and is not very used anyway.
# def test_resnet_200():
#     num_classes = 10
#     input_shape = (20, 224, 224, 3)

#     model = maskflow.model.ResNet200(num_classes=num_classes, include_top=True, data_format='channels_last')
#     model.build(input_shape=input_shape)

#     print(model.count_params())
#     assert model.count_params() == 62886026

#     output_shape = model.compute_output_shape(input_shape)
#     print(output_shape)
#     npt.assert_equal(output_shape, [20, 10])
