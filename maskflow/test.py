from pathlib import Path
import logging

import maskflow


def run_all_tests():

    try:
        import pytest
    except ImportError:
        logging.error("You need to install pytest to run tests.")
        return

    maskflow_dir = Path(maskflow.__file__).parent

    if maskflow_dir.is_dir():
        pytest.main(["-v", str(maskflow_dir)])
    else:
        mess = f"maskflow directory can't be found: {maskflow_dir}"
        logging.error(mess)
