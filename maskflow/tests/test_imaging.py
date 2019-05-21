import numpy as np
import maskflow


def test_encode_image():
    image = np.zeros((128, 128), dtype="uint8")
    image[20:40, 55:95] = 255

    encoded_image = maskflow.imaging.encode_image(image, image_format="tiff")
    assert encoded_image[:10] == "SUkqAAgAAA"

    encoded_image = maskflow.imaging.encode_image(image, image_format="jpeg")
    assert encoded_image[:10] == "/9j/4AAQSk"

    encoded_image = maskflow.imaging.encode_image(image, image_format="png")
    assert encoded_image[:10] == "iVBORw0KGg"
