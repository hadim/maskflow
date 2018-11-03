import datetime
import logging

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from .config import save_config


def setup_logging(log_path):
    '''Set logging.
    '''
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)

    # Disable some verbose loggers
    logging.getLogger("maskrcnn_benchmark.utils.model_serialization").setLevel(logging.WARN)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    

def build_model(config, model_dir, use_last_model, model_to_use,
                distributed=False, local_rank=0):
    '''Setup directory for training and build the model.
    
    Args:
        config: dict, configuration of the model.
        model_dir: Path, parent folder of model's directories.
        use_last_model: bool, use the last model in `model_dir`.
        use_last_model: str, name of the folder in `model_dir` to restore checkpoints.
    '''
    
    # Configure folder to save model checkpoints and log files.
    if model_to_use:
        current_model_path = model_dir / model_to_use
    else:
        if use_last_model:
            folders = sorted(list(filter(lambda p: p.is_dir(), model_dir.iterdir())))
            # If it exists select the most recent folder.
            assert len(folders) >= 1, f'{model_dir} does not contain any directories.'
            model_to_use = folders[-1]
            current_model_path = model_dir / model_to_use
        else:
            now = datetime.datetime.now()
            model_to_use = now.strftime("%Y.%m.%d-%H:%M:%S")
            current_model_path = model_dir / model_to_use
            current_model_path.mkdir(exist_ok=True)

    assert current_model_path.is_dir(), f'{current_model_path} does not exist'

    # Configure logging
    log_path = current_model_path / 'training.log'
    setup_logging(log_path)

    logging.info(f'Training model directory set: {current_model_path}')
    
    # Set the model dir in the configuration.
    config['OUTPUT_DIR'] = str(current_model_path)

    logging.info(f'Building the model...')
    
    # Build the model
    model = build_detection_model(config)
    device = torch.device(config['MODEL']['DEVICE'])
    model.to(device)

    # Create optimizer and scheduler
    optimizer = make_optimizer(config, model)
    scheduler = make_lr_scheduler(config, optimizer)

    # Set distributed if needed
    if distributed:
        torch.cuda.set_device(local_rank)
        model = torch.nn.parallel.deprecated.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,)

    # Configure checkpoints and load previous weights.
    save_to_disk = local_rank == 0
    checkpointer = DetectronCheckpointer(config, model, optimizer, scheduler, config['OUTPUT_DIR'] , save_to_disk)
    extra_checkpoint_data = checkpointer.load(config['MODEL']['WEIGHT'])

    # Create the data loader instance.
    logging.info(f'Create the data loader.')
    data_loader = make_data_loader(config, is_train=True, is_distributed=distributed, start_iter=0)

    # Do some checking
    num_classes_error = "The number of classes in the dataset MUST match the one in the configuration file minus one (background)."
    assert len(data_loader.dataset.coco.cats) + 1 == config['MODEL']['ROI_BOX_HEAD']['NUM_CLASSES'], num_classes_error
    
    training_args = {}
    training_args['model'] = model
    training_args['data_loader'] = data_loader
    training_args['optimizer'] = optimizer
    training_args['scheduler'] = scheduler
    training_args['checkpointer'] = checkpointer
    training_args['device'] = device
    training_args['checkpoint_period'] = config['SOLVER']['CHECKPOINT_PERIOD']
    training_args['arguments'] = extra_checkpoint_data if extra_checkpoint_data else {'iteration': 0}
    
    # Save the configuration used in the model folder.
    save_config(config, current_model_path / "config.yaml")
    
    logging.info('Model ready to be use.')
    
    return training_args, current_model_path
