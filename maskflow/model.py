from pathlib import Path
import tempfile
import zipfile
import shutil

import torch


def export_model(model, model_path, exported_model_dir, exported_name):
    dir_path = Path(tempfile.mkdtemp())

    # Copy files to temp folder
    torch.save(model, dir_path / "model.pth")
    shutil.copy(model_path / 'config.yaml', dir_path)
    shutil.copy(model_path / 'training.log', dir_path)
    shutil.copy(model_path / 'training_metrics.csv', dir_path)
    shutil.copy(model_path / 'evaluation.json', dir_path)

    # Create archive
    fname = shutil.make_archive(exported_model_dir / exported_name, 'zip', root_dir=dir_path, base_dir=None)

    # Cleanup
    shutil.rmtree(dir_path)
    
    return fname