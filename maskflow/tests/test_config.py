from pathlib import Path
import tempfile

import maskflow


def test_update_config():
    config = maskflow.get_default_config()
    new_config = {"DATASET": {"IMAGE_SIZE": 512}}
    config = maskflow.update_config(config, new_config)
    print(config)
    assert config["DATASET"]["IMAGE_SIZE"] == 512


def test_get_default_config():
    config = maskflow.get_default_config()
    print(config)

    assert "DATASET" in config.keys()
    assert "CLASS_NAMES" in config["DATASET"].keys()
    assert config["DATASET"]["IMAGE_SIZE"] == 128


def test_config_io():
    config = maskflow.get_default_config()

    _, config_path = tempfile.mkstemp()
    maskflow.save_config(config, config_path)

    assert Path(config_path).exists()

    config = maskflow.load_config(config_path)
    assert "DATASET" in config.keys()
    assert config["DATASET"]["IMAGE_SIZE"] == 128
