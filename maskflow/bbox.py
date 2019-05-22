import numpy as np


def get_bbox(mask):
    xx, yy = np.argwhere(mask == True).T

    x1 = xx.min()
    x2 = xx.max()
    y1 = yy.min()
    y2 = yy.max()
    w = x2 - x1
    h = y2 - y1
    return (x1, y1), w, h