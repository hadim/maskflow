import colorsys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
import tqdm
from skimage.measure import find_contours

from ipywidgets import widgets
from ipywidgets import interact
from ipywidgets import fixed


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def show_images(images_list, size=14, cmap="viridis", use_widget=True, t=0):

    if isinstance(images_list, list):
        images_list = np.array(images_list)

    for i, images in enumerate(images_list):
        if images.ndim == 3 and images.shape[-1] <= 3:
            images = [images]
        elif images.ndim == 2:
            images = [images]
        images_list[i] = np.array(images)
        
    axs = get_ax(rows=1, cols=len(images_list), size=size)
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    else:
        axs = axs.flat
        
    fig = axs[0].get_figure()
    plt.close(fig)

    def show(t):   
        for i, ax in enumerate(axs):
            ax.clear()
            ax.imshow(images_list[i][t], cmap=cmap)
            
        fig = axs[0].get_figure()
        display(fig)

    if use_widget:
        interact(show, t=widgets.IntSlider(value=0, min=0, max=len(images_list[0]) - 1))
    else:
        show(t=t)
    
    
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    colors = np.array(colors) * 255
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def get_masked_fixed_color(image, boxes, masks, class_ids, class_names,
                           colors=None, scores=None, title="",
                           draw_boxes=False, draw_masks=False,
                           draw_contours=False, draw_score=False):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    if colors is None:
        colors = random_colors(len(class_names))

    masked_image = np.array(image)

    for i in range(N):
        color = colors[class_ids[i]]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if draw_boxes:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, thickness=1)

        # Label
        if draw_score:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = np.random.randint(x1, (x1 + x2) // 2)
            caption = f"{score:.3f}" if score else label
            cv2.putText(masked_image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color)

        # Mask
        mask = masks[:, :, i]
        if draw_masks:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if draw_contours:
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                verts = verts.reshape((-1, 1, 2)).astype(np.int32)
                # Draw an edge on object contour
                cv2.polylines(masked_image, verts, True, color)

    return masked_image


def draw_objects(image, results, class_names,
                 resize_ratio=1, colors=None,
                 draw_boxes=False, draw_masks=True,
                 draw_contours=True, draw_score=True):
    
    if isinstance(image, list):
        image = np.array(image)
    
    if image.ndim == 3 and image.shape[-1] <= 3:
        image = np.array([image])
    elif image.ndim == 2:
        image = np.array([image])
        
    if isinstance(results, dict):
        results = [results]
    
    masked_image_batch = []
    
    if not colors:
        colors = random_colors(len(class_names))

    n_results = len(results)
    for i in tqdm.tqdm(range(n_results), total=n_results):
        r = results[i]
        im = image[i]
        
        # Remove background ids
        class_ids = r['class_ids'] - 1
        
        if "scores" in r:
            scores = r['scores']
        else:
            scores = None
        masked_image = get_masked_fixed_color(im, r['rois'], r['masks'], class_ids,
                                              class_names, colors, scores,
                                              draw_boxes=draw_boxes, draw_masks=draw_masks,
                                              draw_contours=draw_contours, draw_score=draw_score)
        masked_image = cv2.resize(masked_image, None, interpolation=cv2.INTER_NEAREST,
                                  fx=resize_ratio, fy=resize_ratio)
        masked_image_batch.append(masked_image)

    masked_image_batch = np.array(masked_image_batch)
    return masked_image_batch
