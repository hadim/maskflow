from pathlib import Path
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
from . import model
from . import train


def setup_logging(log_path=None):
  """Set logging.
  """

  # Reset previous loggers
  for h in logging.getLogger().handlers:
    logging.root.removeHandler(h)
  for f in logging.getLogger().filters:
    logging.root.removeFilter(f)

  formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.INFO)

  console_handler = logging.StreamHandler()
  console_handler.setFormatter(formatter)
  root_logger.addHandler(console_handler)

  # Log to a file.
  if log_path:
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


setup_logging()


def run_all_tests():

  try:
    import pytest
  except ImportError:
    logging.error("You need to install pytest to run tests.")
    return

  maskflow_dir = Path(__file__).parent

  if maskflow_dir.is_dir():
    pytest.main(["-v", str(maskflow_dir)])
  else:
    mess = f"maskflow directory can't be found: {maskflow_dir}"
    logging.error(mess)
