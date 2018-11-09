import logging

from . import config
from . import dataset
from . import cococreator
from . import viz
from . import training
from . import inference
from . import utils
from . import model
from . import archive

__all__ = [config, dataset, cococreator, viz, training, inference, utils, model, archive]


def setup_logging():
    '''Set logging.
    '''
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Disable some verbose loggers
    logging.getLogger("maskrcnn_benchmark.utils.model_serialization").setLevel(logging.WARN)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

setup_logging()