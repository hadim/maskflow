import numpy as np


def from_mask(mask):
    xx, yy = np.argwhere(mask > 0).T

    x1 = xx.min()
    x2 = xx.max()
    y1 = yy.min()
    y2 = yy.max()
    w = x2 - x1
    h = y2 - y1
    return np.array([x1, y1, w, h]).astype("float")


def from_masks(masks):
    bboxes = []
    for mask in masks:
        bboxes.append(from_mask(mask))
    return np.array(bboxes).astype("float")
