import matplotlib.pyplot as plt
import numpy as np

def display_top_masks(image, mask, class_ids, class_names, basesize=14, limit=4, cmap="Blues_r"):
    """Display the given images and the top few class masks."""
    
    to_display = []
    titles = []
    to_display.append(image)
    
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    
    # Remove 0 class ids
    idx_to_keep = class_ids > 0
    class_ids = class_ids[idx_to_keep]
    mask = mask[idx_to_keep]
    
    unique_class_ids = np.unique(class_ids)

    n_ids = limit
    if limit > len(unique_class_ids):
        n_ids = len(unique_class_ids)
    
    # Generate images and titles
    for i in range(n_ids):
        class_id = unique_class_ids[i]
        
        # Pull masks of instances belonging to the same class.
        m = mask[np.where(class_ids == class_id)[0]]
        
        # We use a multiplier to differentiate between different instances.
        multipliers = np.arange(1, m.shape[0] + 1)[:, np.newaxis, np.newaxis]
        m = np.sum(m * multipliers, axis=0)

        to_display.append(m)
        titles.append(class_names[class_id - 1] if class_id != -1 else "-")
        
    return display_images(to_display, titles=titles, cols=limit + 1, basesize=basesize, cmap=cmap)
    
    
def display_images(images, titles=None, cols=4, basesize=14, cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols
    rows = rows if len(images) % cols == 0 else rows + 1
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(basesize, basesize * rows // cols))
    for image, title, ax in zip(images, titles, axs.flatten()):
        ax.set_title(title, fontsize=9)
        ax.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
    return fig


def batch_display_top_masks(images, masks, class_ids, class_names, basesize=14, limit=4, cmap="Blues_r"):
    for image, mask, single_class_ids in zip(images, masks, class_ids):
        display_top_masks(image, mask, single_class_ids, class_names, basesize=basesize, limit=limit, cmap=cmap)