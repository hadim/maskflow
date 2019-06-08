import matplotlib.pyplot as plt
import numpy as np
import cv2

from . import imaging


def compute_colors_for_labels(labels):
  """Simple function that adds fixed colors depending on the class
  """
  palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1])
  colors = labels[:, None] * palette
  colors = (colors % 255).astype("float")
  colors /= 255
  colors[:, -1] = 1
  return colors


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

  fig, axs = plt.subplots(ncols=cols, nrows=rows,
                          figsize=figsize, constrained_layout=True)
  for image, title, ax in zip(images, titles, axs.flatten()):
    ax.set_title(title, fontsize=9)
    ax.imshow(image, cmap=cmap, norm=norm,
              interpolation=interpolation, origin=[0, 0])

  return fig


def display_top_masks(image, masks, label_ids, class_names, basesize=14, limit=4, cmap="PuBu_r"):
  """Display the given images and the top few class masks.

  Args:
      image: The image as an array of shape (W, H, C).
      mask: The instance masks of the different objects.
      label_ids: The label IDs of the instance masks in the same order as `masks`.
      class_names: The name of the class.
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
  unique_labels, counts = np.unique(label_ids, return_counts=True)
  sorted_indices = np.argsort(-counts)
  unique_labels = unique_labels[sorted_indices]
  unique_labels = unique_labels[:limit]

  for label_id in unique_labels:

    # Get all the masks for this label id
    all_masks = [masks[i] for i in np.where(label_id == label_ids)[0]]
    all_masks = np.array(all_masks)

    # We use a multiplier to differentiate between different instances.
    multipliers = np.arange(
        1, all_masks.shape[0] + 1)[:, np.newaxis, np.newaxis]
    merged_masks = np.sum(all_masks * multipliers, axis=0)

    to_display.append(merged_masks)

    label_name = class_names[label_id]
    titles.append(f"Label: {label_name} ({all_masks.shape})")

  return display_images(to_display, titles=titles, cols=limit + 1, basesize=basesize, cmap=cmap)


def batch_display_top_masks(dataset, class_names, basesize=14, limit=4, cmap="PuBu_r"):
  """Display the given images and the top few class masks. The input is a Maskflow `tf.data.Dataset`.

  See `maskflow.viz.display_top_masks`.
  """
  for feature in dataset:

    image = feature['image'].numpy()
    masks = feature['masks'].numpy()
    bboxes = feature['bboxes'].numpy()
    label_ids = feature['label_ids'].numpy()
    label_names = feature['label_names'].numpy()

    # Remove padded values.
    to_keep = label_ids > -1
    masks = masks[to_keep]
    bboxes = bboxes[to_keep]
    label_ids = label_ids[to_keep]
    label_names = label_names[to_keep]

    display_top_masks(image, masks, label_ids, class_names,
                      basesize=basesize, limit=limit, cmap=cmap)


def draw_image(image,
               masks=None,
               bboxes=None,
               label_names=None,
               label_ids=None,
               class_names=None,
               draw_bbox=True,
               draw_true_label=False,
               draw_mask=True,
               draw_contour=True,
               rectangle_line_thickness=1,
               text_line_thickness=1,
               mask_alpha=0.2,
               contour_line_thickness=2):
  """Overlay an image with detected objects as masks, bounding boxes or contours.

  Args:
      TODO
  """

  colors = compute_colors_for_labels(np.arange(len(class_names)))
  drawn_image = image.copy()

  for i in enumerate(label_ids):

    label_id = label_ids[i]
    color = colors[label_id] * 255

    if draw_bbox:
      bbox = bboxes[i]
      point1 = tuple(bbox[:2])[::-1]
      point2 = tuple(bbox[:2] + bbox[2:])[::-1]
      drawn_image = cv2.rectangle(
          drawn_image, point1, point2, color=color, thickness=rectangle_line_thickness)

    if draw_true_label:
      label_name = label_names[i].decode('utf8')
      text = f"{label_name} ({label_id})"
      x = int(bbox[1] - 5)
      y = int(bbox[0])
      drawn_image = cv2.putText(drawn_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.2, color=color, thickness=text_line_thickness)

    if draw_contour:
      mask = masks[i]
      contours, _ = cv2.findContours(
          mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      drawn_image = cv2.drawContours(
          drawn_image, contours, -1, color, thickness=contour_line_thickness)

  if draw_mask:
    drawn_image = imaging.blend_image_with_masks(
        image, masks, colors, alpha=mask_alpha)

  return drawn_image


def draw_dataset(dataset,
                 class_names=None,
                 draw_bbox=True,
                 draw_true_label=False,
                 draw_mask=True,
                 draw_contour=True,
                 rectangle_line_thickness=1,
                 text_line_thickness=1,
                 mask_alpha=0.2,
                 contour_line_thickness=2):

  overlays = []

  for feature in dataset:
    image = feature['image'].numpy()
    masks = feature['masks'].numpy()
    bboxes = feature['bboxes'].numpy()
    label_ids = feature['label_ids'].numpy()
    label_names = feature['label_names'].numpy()

    to_keep = label_ids > -1
    masks = masks[to_keep]
    bboxes = bboxes[to_keep]
    label_ids = label_ids[to_keep]
    label_names = label_names[to_keep]

    draw_args = {}
    draw_args['image'] = image
    draw_args['masks'] = masks
    draw_args['bboxes'] = bboxes
    draw_args['label_names'] = label_names
    draw_args['label_ids'] = label_ids
    draw_args['class_names'] = class_names
    draw_args['draw_bbox'] = draw_bbox
    draw_args['draw_true_label'] = draw_true_label
    draw_args['draw_mask'] = draw_mask
    draw_args['draw_contour'] = draw_contour
    draw_args['rectangle_line_thickness'] = rectangle_line_thickness
    draw_args['text_line_thickness'] = text_line_thickness
    draw_args['mask_alpha'] = mask_alpha
    draw_args['contour_line_thickness'] = contour_line_thickness

    overlays.append(draw_image(**draw_args))

  return overlays
