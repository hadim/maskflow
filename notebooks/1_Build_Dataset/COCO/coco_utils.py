import urllib
import tempfile
import tqdm
import os
import zipfile


class TqdmUpTo(tqdm.tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

        
def download_cocodataset(extract_folder_path):
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    temp_path = tempfile.mktemp(suffix=".zip", prefix="coco_dataset_")

    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urllib.request.urlretrieve(annotations_url, filename=temp_path, reporthook=t.update_to, data=None)

    with zipfile.ZipFile(temp_path) as zf:
        zf.extractall(extract_folder_path)

    os.remove(temp_path)
    
