import logging

from ._version import __version__

from .config import update_config
from .config import get_default_config
from .config import load_config
from .config import save_config

from . import dataset
from . import image

#from . import dataset
#from . import cococreator
#from . import viz
#from . import training
#from . import inference
#from . import utils
#from . import model
#from . import archive


def setup_logging():
    """Set logging.
    """
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

setup_logging()
