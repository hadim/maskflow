import logging

from ._version import __version__

from .config import update_config
from .config import get_default_config
from .config import load_config
from .config import save_config

from . import dataset
from . import imaging
from . import viz
from . import utils
from . import mask
from . import bbox


def setup_logging():
    """Set logging.
    """
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # # Log to a file.
    # file_handler = logging.FileHandler(log_path)
    # file_handler.setFormatter(formatter)
    # root_logger.addHandler(file_handler)

setup_logging()
