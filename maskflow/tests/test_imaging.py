import numpy as np
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
