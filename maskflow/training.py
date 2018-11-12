import datetime
import logging
import time

import torch
import pandas as pd
import numpy as np

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
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict

from .config import save_config
from . import utils


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

    
def _get_pretrained_weights_url(config):
    base_pretrained_weights_url = "https://download.pytorch.org/models/maskrcnn/e2e_{}_rcnn_{}_1x.pth"
    body_name = config['MODEL']['BACKBONE']['CONV_BODY']
    body_name = body_name.replace('-', '_')
    net_head_name = 'mask' if config['MODEL']['MASK_ON'] else 'faster'
    return base_pretrained_weights_url.format(net_head_name, body_name)


def _transfer_pretrained_weights(model, pretrained_model_pth):
    # Load the weights 
    pretrained_weights = torch.load(pretrained_model_pth)['model']

    # Remove `module` from the keys and remove the weights that depends on the number of classes.
    pretrained_weights = {k.replace('module.',''): v for k, v in pretrained_weights.items()
                if 'cls_score' not in k and 'bbox_pred' not in k and 'mask_fcn_logits' not in k}

    # Feed the model with the weights
    this_state = model.state_dict()
    this_state.update(pretrained_weights)
    model.load_state_dict(this_state)


def build_model(config, model_dir, use_last_model, model_to_use,
                distributed=False, use_pretrained_weights=True):
    '''Setup directory for training and build the model.
    
    Args:
        config: dict, configuration of the model.
        model_dir: Path, parent folder of model's directories.
        use_last_model: bool, use the last model in `model_dir`.
        use_last_model: str, name of the folder in `model_dir` to restore checkpoints.
        use_pretrained_weights: bool, use pretrained weights if possible.
    '''
    
    # Do some configuration checks.

    pooler_resolution = config['MODEL']['ROI_MASK_HEAD']['RESOLUTION'] // 2
    config['MODEL']['ROI_MASK_HEAD']['POOLER_RESOLUTION'] = pooler_resolution
    
    # Configure folder to save model checkpoints and log files.
    current_model_path = None
    is_a_new_model = False
    if model_to_use:
        current_model_path = model_dir / model_to_use
    else:
        if use_last_model:
            folders = sorted(list(filter(lambda p: p.is_dir(), model_dir.iterdir())))
            if len(folders) >= 1:
                model_to_use = folders[-1]
                current_model_path = model_dir / model_to_use
    
    if not current_model_path:
        now = datetime.datetime.now()
        model_to_use = now.strftime("%Y.%m.%d-%H:%M:%S")
        current_model_path = model_dir / model_to_use
        current_model_path.mkdir(exist_ok=True)
        is_a_new_model = True

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
    
    if is_a_new_model and use_pretrained_weights:
        # Get the URL of the pretrained weights
        pretrained_weights_url = _get_pretrained_weights_url(config)

        # Define the loc path of the pretrained weights
        pretrained_weights_path = model_dir / "pretrained_weights_coco.pth"

        # Download pretrained weights if necessary
        utils.download_file(pretrained_weights_url, pretrained_weights_path)
        logging.info(f"Use pretrained weights from {pretrained_weights_url}")

        # Load weights to the model
        _transfer_pretrained_weights(model, pretrained_weights_path)

    # Create optimizer and scheduler
    optimizer = make_optimizer(config, model)
    scheduler = make_lr_scheduler(config, optimizer)

    # Set distributed if needed
    if distributed:
        # Does not work.
        model = torch.nn.DataParallel(model)

    # Configure checkpoints and load previous weights.
    save_to_disk = True
    checkpointer = DetectronCheckpointer(config, model, optimizer, scheduler, config['OUTPUT_DIR'], save_to_disk)
    extra_checkpoint_data = checkpointer.load(config['MODEL']['WEIGHT'])

    if is_a_new_model:
        checkpointer.save('model_0000000')
    
    # Create the data loader instance.
    logging.info(f'Create the data loader.')
    data_loader = make_data_loader(config, is_train=True, is_distributed=False, start_iter=0)

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
    training_args['model_path'] = current_model_path
    
    # Save the configuration used in the model folder.
    save_config(config, current_model_path / "config.yaml")
    
    logging.info('Model ready to be trained.')
    
    return training_args


def do_train(model, data_loader, optimizer, scheduler, checkpointer,
             device, checkpoint_period, arguments, model_path,
             log_period=20, log_losses_detailed=False, save_metrics=True,
             tensorboard=True, tensorboard_log_period=20):
    
    logger = logging.getLogger("maskfow.training")
    logger.info(f"Start training at iteration {arguments['iteration']}")
                  
    if tensorboard:
        from tensorboardX import SummaryWriter
        log_path = model_path / 'logs'
        writer = SummaryWriter(log_dir=str(log_path))
        #writer.add_graph(model)
        logger.info(f"tensorboard --logdir {log_path}")
                
    meters = MetricLogger(delimiter="  ")
                
    max_iter = len(data_loader)
    start_iter = arguments['iteration']
    model.train()
                
    logger.info(f"Training will stop at {max_iter}")
                
    start_training_time = time.time()
    end = time.time()
                
    # Record losses
    metrics = pd.DataFrame([])
    metrics_path = model_path / "training_metrics.csv"
                
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
                
        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = np.sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = np.sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
        datum = {}
        datum.update({k: v.to('cpu').detach().numpy() for k, v in loss_dict_reduced.items()})
        datum['loss'] = losses_reduced.to('cpu').detach().numpy()
        datum['eta'] = eta_string
        datum['iteration'] = iteration
        datum['memory'] = torch.cuda.max_memory_allocated() / float(1<<20)
        datum['lr'] = optimizer.param_groups[0]["lr"]
        metrics = metrics.append(pd.Series(datum), ignore_index=True)
                
        if iteration % log_period == 0 or iteration == (max_iter - 1):
            log_str = "Step: {iteration} | Loss: {loss:.6f} | ETA: {eta} | LR: {lr:.6f} | Memory: {memory:.0f} MB"
            log_str = log_str.format(**datum)
            logger.info(log_str)
            if log_losses_detailed:
                logger.info(str(meters))
                
        if iteration % tensorboard_log_period == 0 or iteration == (max_iter - 1):
            if tensorboard:
                writer.add_scalar(f'loss/average_loss', metrics['loss'].mean(), global_step=iteration)
                writer.add_scalar(f'loss/loss', datum['loss'], global_step=iteration)
                for key, value in loss_dict_reduced.items():
                    writer.add_scalar(f'loss/{key}', value.clone().to('cpu').detach().numpy(), global_step=iteration)
                writer.add_scalar('general/memory', datum['memory'], global_step=iteration)
                writer.add_scalar('general/lr', datum['lr'], global_step=iteration)
                
        if iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer_args = {'iteration': iteration}
            checkpointer.save("model_{:07d}".format(iteration), **checkpointer_args)
                
        if save_metrics:
            metrics.to_csv(metrics_path)
                
    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (max_iter)))
                
    if tensorboard:
        writer.close()
