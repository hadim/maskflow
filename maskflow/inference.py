import json

import numpy as np
import torch

from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from .dataset import get_data_loader


def build_model(config, model_path):
    model = build_detection_model(config)
    model.to(config['MODEL']['DEVICE'])

    # Load last checkpoint
    last_checkpoints = sorted(model_path.glob('*.pth'))
    assert len(last_checkpoints) >= 1, f"No checkpoint to load in {model_path}"
    last_checkpoint = last_checkpoints[-1]
    _ = DetectronCheckpointer(config, model).load(str(last_checkpoint))
    
    model.eval()
    return model


def run_evaluation(config, model_path, data_dir):
    
    model = build_model(config, model_path)
    data_loader = get_data_loader(config, data_dir, is_train=False)

    inference_args = {}
    inference_args['model'] = model
    inference_args['data_loader'] = data_loader
    inference_args['iou_types'] = ("bbox", "segm")
    inference_args['box_only'] = config['MODEL']['RPN_ONLY']
    inference_args['device'] = config['MODEL']['DEVICE']
    inference_args['expected_results'] = ()
    inference_args['expected_results_sigma_tol'] = 4
    inference_args['output_folder'] = None

    # Run evaluation
    results, coco_result, predictions = inference(**inference_args)

    # Save results
    evaluation_path = model_path / "evaluation.json"
    with open(evaluation_path, 'w') as f:
        json.dump(dict(results.results), f, indent=2)
        
    return results


def merge_mask(mask):
    multipliers = np.arange(1, mask.shape[0] + 1)[:, np.newaxis, np.newaxis]
    return np.sum(mask * multipliers, axis=0)


def get_bbox(mask):
    xx, yy = np.argwhere(mask == True).T

    x1 = xx.min()
    x2 = xx.max()
    y1 = yy.min()
    y2 = yy.max()
    w = x2 - x1
    h = y2 - y1
    return (x1, y1), w, h


def select_top_predictions(predictions, confidence_threshold):
        """
        Select only predictions which have a `score` > confidence_threshold,
        and returns the predictions in descending order of score
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]


def compute_colors_for_labels(labels):
    """Simple function that adds fixed colors depending on the class
    """
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).astype("float")
    colors /= 255
    colors[:, -1] = 1
    return colors
