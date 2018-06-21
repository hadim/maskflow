import pandas as pd
import numpy as np

import skimage
import tifffile
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

from mrcnn import utils
from mrcnn import visualize


class MaskflowDataset(utils.Dataset):

    def set_dataset(self, fnames, class_names):
        """
        :fnames: a list of Python pathlib.Path objects.
        :class_names: a list of str.
        """
        
        source_name = ""
        
        # Add classes
        for i, class_name in enumerate(class_names):
            self.add_class(source_name, i+1, class_name)

        # Add image specifications
        for i, fname in enumerate(fnames):

            image_info = {}
            image_info["source"] = source_name
            image_info["image_id"] = i
            image_info["path"] = fname
            
            mask_path = fname.parent.parent / "Mask" / fname.name
            image_info["mask_path"] = mask_path
            
            class_ids_path = fname.parent.parent / "Class" / fname.with_suffix(".csv").name
            image_info["class_ids_path"] = class_ids_path
            
            assert fname.is_file(), f"Image file {fname} doesn't exist."
            assert mask_path.is_file(), f"Mask image file {mask_path} doesn't exist."
            assert class_ids_path.is_file(), f"Class ids image file {class_ids_path} doesn't exist."
            
            self.add_image(**image_info)
            
        self.prepare()

    def load_image(self, image_id):
        
        info = self.image_info[image_id]
        im = tifffile.imread(str(info["path"]))
        
        # Convert to 8bit
        im = skimage.util.img_as_ubyte(im)
        im = skimage.exposure.rescale_intensity(im)
        
        # Convert to RGB
        im = skimage.color.grey2rgb(im)
        
        return im

    def load_mask(self, image_id):

        info = self.image_info[image_id]

        # Open masks
        mask = tifffile.imread(str(info["mask_path"]))
        #mask = np.swapaxes(mask, 2, 0)
        count = mask.shape[-1]
        
        # Handle occlusions
        handle_occlusion = True
        if handle_occlusion:
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Open class ids file.
        class_ids = pd.read_csv(info["class_ids_path"], header=None).values[:, 0]

        return mask.astype(np.bool), class_ids.astype(np.int32)

    def random_display(self, n=4, n_class=4):
        # Load and display random samples
        image_ids = np.random.choice(self.image_ids, n, replace=True)
        
        for image_id in image_ids:
            print(self.image_info[image_id])
            image = self.load_image(image_id)
            mask, class_ids = self.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, self.class_names, limit=n_class)
            