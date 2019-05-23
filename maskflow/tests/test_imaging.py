import numpy as np
import numpy.testing as npt
import scipy.misc

import maskflow


def test_encode_image():
    image = np.zeros((128, 128), dtype="uint8")
    image[20:40, 55:95] = 255

    # encoded_image = maskflow.imaging.encode_image(image, image_format="tiff")
    # assert encoded_image[:10] == "SUkqAAgAAA"

    encoded_image = maskflow.imaging.encode_image(image, image_format="jpeg")
    assert encoded_image.numpy()[:10] == b'\xff\xd8\xff\xe0\x00\x10JFIF'

    encoded_image = maskflow.imaging.encode_image(image, image_format="png")
    assert encoded_image.numpy()[:10] == b'\x89PNG\r\n\x1a\n\x00\x00'


def test_blend_image():
    # Test on RGB image.
    image = scipy.misc.face(gray=False)

    alpha = 0.4
    colors = [(1, 0, 0), (0, 1, 0)]

    masks = []
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask[200:500, 300:600] = 1
    masks.append(mask)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask[450:700, 500:70000] = 1
    masks.append(mask)

    image_blended = maskflow.imaging.blend_image_with_masks(image, masks,
                                                            colors, alpha=alpha)

    npt.assert_almost_equal(image_blended.mean(), 107.6187761094835)
    npt.assert_almost_equal(image_blended.std(), 57.54396217773932)
    npt.assert_almost_equal(image_blended.sum(), 253904548)

    # Test on gray image.
    image = scipy.misc.face(gray=True)

    alpha = 0.4
    colors = [(1, 0, 0), (0, 1, 0)]

    masks = []
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask[200:500, 300:600] = 1
    masks.append(mask)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask[450:700, 500:70000] = 1
    masks.append(mask)

    image_blended = maskflow.imaging.blend_image_with_masks(image, masks,
                                                            colors, alpha=alpha)

    npt.assert_almost_equal(image_blended.mean(), 110.70499886406793)
    npt.assert_almost_equal(image_blended.std(), 54.80564959813203)
    npt.assert_almost_equal(image_blended.sum(), 261185861)
