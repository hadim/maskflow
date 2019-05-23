import tensorflow as tf

from . import pad_first_dimension


def _pad_dataset(max_num_instances):
    """Pad tensors using the parameter `MAX_NUM_INSTANCES`.
    """
    def _fn(feature):
        num_elements = max_num_instances
        feature['bboxes'] = pad_first_dimension(feature['bboxes'], padded_value=0, num_elements=num_elements)
        feature['label_ids'] = pad_first_dimension(feature['label_ids'], padded_value=-1, num_elements=num_elements)
        feature['label_names'] = pad_first_dimension(feature['label_names'], padded_value="", num_elements=num_elements)
        feature['masks'] = pad_first_dimension(feature['masks'], padded_value=0, num_elements=num_elements)

        return feature
    return _fn


def preprocess_dataset(dataset, max_num_instances):
    """Preprocess a Maskflow dataset.

    - Padding values using `max_num_instances` bboxes, label_ids,
      label_names and masks.

    Args:
        dataset: `tf.data.Dataset`, a dataset.
        max_num_instances: `int`, Number to pad data with.

    Returns:
        tf.data.Dataset
    """

    _pad_dataset_fn = _pad_dataset(max_num_instances)
    dataset = dataset.map(_pad_dataset_fn)

    return dataset
