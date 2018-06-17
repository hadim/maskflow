import numpy as np
import cv2

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def random_shape(height, width, class_names):
    """Generates specifications of a random shape that lies within
    the given height and width boundaries.
    Returns a tuple of three valus:
    * The shape name (square, circle, ...)
    * Shape color: a tuple of 3 values, RGB.
    * Shape dimensions: A tuple of values that define the shape size
                        and location. Differs per shape type.
    """
    # Shape
    shape = np.random.choice(class_names[1:])
    # Color
    color = tuple([np.random.randint(0, 255) for _ in range(3)])
    # Center x, y
    buffer = 20
    y = np.random.randint(buffer, height - buffer - 1)
    x = np.random.randint(buffer, width - buffer - 1)
    # Size
    s = np.random.randint(buffer, height // 4)
    return shape, color, (x, y, s)


def random_image( height, width, max_n, class_names):
    """Creates random specifications of an image with multiple shapes.
    Returns the background color of the image and a list of shape
    specifications that can be used to draw the image.
    """
    # Pick random background color
    bg_color = np.array([np.random.randint(0, 255) for _ in range(3)])
    # Generate a few random shapes and record their
    # bounding boxes
    shapes = []
    boxes = []
    N = np.random.randint(1, max_n + 1)
    for _ in range(N):
        shape, color, dims = random_shape(height, width, class_names)
        shapes.append((shape, color, dims))
        x, y, s = dims
        boxes.append([y - s, x - s, y + s, x + s])
    # Apply non-max suppression wit 0.3 threshold to avoid
    # shapes covering each other
    keep_ixs = non_max_suppression(np.array(boxes), np.arange(N), 0.3)
    shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
    return bg_color, shapes


def draw_shape(image, shape, dims, color):
    """Draws a shape from the given specs."""
    # Get the center x, y and the size s
    x, y, s = dims
    if shape == 'square':
        image = cv2.rectangle(image, (x - s, y - s),
                              (x + s, y + s), color, -1)
    elif shape == "circle":
        image = cv2.circle(image, (x, y), s, color, -1)
    elif shape == "triangle":
        points = np.array([[(x, y - s),
                            (x - s / np.sin(np.radians(60)), y + s),
                            (x + s / np.sin(np.radians(60)), y + s),
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)
    return image


def generate_image(bg_color, height, width, shapes):
    bg_color = np.array(bg_color).reshape([1, 1, 3])
    image = np.ones([height, width, 3], dtype=np.uint8)
    image = image * bg_color.astype(np.uint8)
    for shape, color, dims in shapes:
        image = draw_shape(image, shape, dims, color)
    return image


def generate_mask(bg_color, height, width, shapes, class_names):
    """
    """
    count = len(shapes)
    mask = np.zeros([height, width, count], dtype=np.uint8)
    for i, (shape, _, dims) in enumerate(shapes):
        mask[:, :, i:i + 1] = draw_shape(mask[:, :, i:i + 1].copy(), shape, dims, 1)
        
    # Handle occlusions
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(count - 2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        
    # Map class names to class IDs.
    class_ids = np.array([class_names.index(s[0]) for s in shapes])
    return mask, class_ids.astype(np.int32)