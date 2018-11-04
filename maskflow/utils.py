from pathlib import Path
import urllib
import tempfile
import logging
import tqdm
import os
import zipfile


class _TqdmUpTo(tqdm.tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Args:
            b: int, optional
                Number of blocks transferred so far [default: 1].
            bsize: int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize: int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

        
def download_zip(zip_url, extract_folder_path, progressbar=True):
    """Download a ZIP file from an URL and extract to a given local folder.
    
    Args:
        zip_url: The URL to the ZIP file as a str.
        extract_folder_path: The path to the local folder for the extraction.
    """
    temp_path = tempfile.mktemp(suffix=".zip")

    with _TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, disable=not progressbar) as t:
        urllib.request.urlretrieve(zip_url, filename=temp_path, reporthook=t.update_to, data=None)

    with zipfile.ZipFile(temp_path) as zf:
        zf.extractall(extract_folder_path)

    os.remove(temp_path)
    

def download_file(file_url, local_file_path, force=False, progressbar=True):
    """Download a file.
    
    Args:
        file_url: string
        local_file_path: string or Path
        force: bool, if False, don't download if the file already exist.
        progressbar: bool, show a progress bar.
    """
    
    if Path(local_file_path).is_file():
        return True
    
    try:
        with _TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, disable=not progressbar) as t:
            urllib.request.urlretrieve(file_url, filename=local_file_path, reporthook=t.update_to, data=None)
    except Exception as e:
        logging.error(f"'{file_url}' cannot be downloaded.")
        logging.error(e)
        
    return True
