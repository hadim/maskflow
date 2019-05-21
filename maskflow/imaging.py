import base64
from io import BytesIO

import numpy as np
from PIL import Image


def encode_image(image, image_format):
    """Convert Numpy image to a string of JPEG, PNG or TIFF."""
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf8")


def crop_image(image, masks, class_ids, final_size):
    """Crop image and mask to final_size if needed. It add zeros values if image is
    smaller and it crop it if the image is bigger than final_size. Remove empty masks
    after crop if needed. Returns None if no object is left after crop.
    """

    assert image.shape[:2] == masks.shape[-2:], "Image and masks need to have the same size."
    assert class_ids.shape[0] == masks.shape[0], "Class ids and masks need to have the same number of objects."

    w, h = image.shape[:2]

    # We do nothing
    if w == final_size and h == final_size:
        return image, masks, class_ids

    crop_w = final_size - w
    crop_h = final_size - h

    new_w = w + crop_w if crop_w > 0 else final_size
    new_h = h + crop_h if crop_h > 0 else final_size

    new_image = np.zeros((new_w, new_h, image.shape[-1]))
    new_image[:w, :h] = image[:new_w, :new_h]
    new_image = new_image.astype(image.dtype)

    new_masks = np.zeros((masks.shape[0], new_w, new_h))
    new_masks[:, :w, :h] = masks[:, :new_w, :new_h]
    new_masks = new_masks.astype(masks.dtype)

    # Check mask that still contain an object.
    to_keep = np.where(new_masks.sum(axis=-1).sum(axis=-1) > 0)[0]
    new_masks = new_masks[to_keep]
    class_ids = class_ids[to_keep]

    if new_masks.shape[0] == 0:
        return None, None, None
    else:
        return new_image, new_masks, class_ids
