import json

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
