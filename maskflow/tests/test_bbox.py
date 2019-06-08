import numpy as np
import numpy.testing as npt

import maskflow


def test_from_masks():
  masks = []

  mask = np.zeros((128, 128), dtype="uint8")
  mask[20:50, 80:90] = 1
  masks.append(mask)

  mask = np.zeros((128, 128), dtype="uint8")
  mask[80:950, 50:80] = 1
  masks.append(mask)

  bboxes = maskflow.bbox.from_masks(masks)

  excepted_bboxes = [[20, 80, 29, 9], [80, 50, 47, 29]]
  npt.assert_equal(bboxes, excepted_bboxes)
