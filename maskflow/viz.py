import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage

from . import inference


def display_top_masks(image, masks, labels, categories, basesize=14, limit=4, cmap="PuBu_r"):
    """Display the given images and the top few class masks.
    
    Args:
        image: The image as an array of shape (W, H, C).
        mask: The instance masks of the different objects.
        labels: The label IDs of the instance masks in the same order as `masks`.
        categories: The name of the categories.
        basesize: Base size of the figure.
        limit: Limit of masks to show.
        cmap: Optional. Color map to use. For example, "Blues".
        
    Returns:
        The image as a Matplotlib figure.
    """
    
    to_display = []
    titles = []
    
    # Set the image to be displayed    
    to_display.append(image)

    # Set title for the image
    titles.append(f"WxHxC = {image.shape[0]}x{image.shape[1]}x{image.shape[2]}")

    # Here we sort the detected labels by the number of time
    # they appear in the detection.
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique_labels = unique_labels[sorted_indices]
    unique_labels = unique_labels[:limit]

    for label_id in unique_labels:

        # Get all the masks for this label id
        all_masks = [masks.polygons[i].convert('mask').numpy() for i in np.where(label_id == labels)[0]]
        all_masks = np.array(all_masks)

        # We use a multiplier to differentiate between different instances.
        multipliers = np.arange(1, all_masks.shape[0] + 1)[:, np.newaxis, np.newaxis]
        merged_masks = np.sum(all_masks * multipliers, axis=0)

        to_display.append(merged_masks)

        label_name = categories[label_id]
        titles.append(f"Label: {label_name['name']} ({all_masks.shape})")

    return display_images(to_display, titles=titles, cols=limit + 1, basesize=basesize, cmap=cmap)
    

def display_images(images, titles=None, cols=4, basesize=14, cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    
    Args:
        images: list or array of image tensors in HWC format.
        titles: optional. A list of titles to display with each image.
        cols: number of images per row
        cmap: Optional. Color map to use. For example, "Blues".
        norm: Optional. A Normalize instance to map values to colors.
        interpolation: Optional. Image interpolation to use for display.
        
    Returns:
        The image as a Matplotlib figure.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols
    rows = rows if len(images) % cols == 0 else rows + 1
    figsize = (basesize, basesize * rows // cols)
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=figsize, constrained_layout=True)
    for image, title, ax in zip(images, titles, axs.flatten()):
        ax.set_title(title, fontsize=9)      
        ax.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation, origin=[0, 0])
    return fig


def batch_display_top_masks(batch_image, batch_target, batch_idx, categories,
                            basesize=14, limit=4, cmap="PuBu_r",
                            pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):
    """Display the given images and the top few class masks.
    
    See `maskflow.display_top_masks`.
    """
    for image, target, idx in zip(batch_image.tensors, batch_target, batch_idx):
        image = image.numpy().copy().swapaxes(0, 2).swapaxes(0, 1)
                      
        # Undo image normalization
        image += np.array(pixel_mean)
        image *= np.array(pixel_std)
        image = image.astype('uint8')
                      
        labels = target.get_field('labels')
        masks = target.get_field('masks')
        
        display_top_masks(image, masks, labels, categories, basesize=basesize, limit=limit, cmap=cmap)
                      
                      
def display_prediction_and_gt(image, prediction, gt, class_names,
                              confidence_threshold=0.7, mask_threshold=0.5,
                              pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]):

    predicted_masks = prediction.get_field('mask').numpy()[:, 0]
    predicted_mask = inference.merge_mask(predicted_masks)

    gt_masks = np.array([m.convert('mask').numpy() for m in gt.get_field('masks').polygons])
    gt_mask = inference.merge_mask(gt_masks)

    image = image.copy().swapaxes(0, 2).swapaxes(0, 1)
    
    # Undo image normalization
    image += np.array(pixel_mean)
    image *= np.array(pixel_std)
    image = image.astype('uint8')

    to_displayed = [image, image, gt_mask, predicted_mask]
    titles = ['Ground Truth', 'Predicted', 'Ground Truth Mask', 'Predicted Mask']

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), constrained_layout=True)
    axs = axs.flatten()

    colors = inference.compute_colors_for_labels(np.arange(0, len(class_names)))

    for image, title, ax in zip(to_displayed, titles, axs):
        ax.set_title(title, fontsize=16)
        ax.imshow(image, interpolation='none', origin=[0, 0])
        ax.set_aspect('equal')

    for (x1, y1, x2, y2), label in zip(prediction.bbox, prediction.get_field('labels').numpy()):
        w = x2 - x1
        h = y2 - y1
        color = colors[label - 1]
        color = 'red'
        #axs[0].add_artist(matplotlib.patches.Rectangle([x1, y1], w, h, fill=None, linewidth=1, edgecolor=color))
        axs[1].add_artist(matplotlib.patches.Rectangle([x1, y1], w, h, fill=None, linewidth=1, edgecolor=color))
        
    return fig