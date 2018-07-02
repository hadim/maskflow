import datetime
import os
from pathlib import Path
import zipfile
import shutil
import tempfile
import re
import multiprocessing

import numpy as np
import skimage

import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util

import tqdm

import mrcnn
from mrcnn.model import MaskRCNN
from mrcnn.model import data_generator
from mrcnn import model as mrcnn_model

from . import processing_graph 


def split_as_batches(images, config):
    
    images = np.array(images)
    
    image_batches = []
    config.BATCH_SIZE = config.IMAGES_PER_GPU
    n_batches = max(1, images.shape[0] // config.BATCH_SIZE + 0)

    for i in range(n_batches):
        n = i * config.BATCH_SIZE
        image_batch = images[n:n + config.BATCH_SIZE]
        image_batches.append(image_batch)
        
    return image_batches


def predict(image, model, progress=False, verbose=0):
    
    if image.ndim == 3:
        image = np.array([image])
    
    # Split image in multiple batches
    image_batches = split_as_batches(image, model.config)

    # Run Prediction on Batches
    results = []
    for image_batch in tqdm.tqdm(image_batches, disable=not progress):
        results.extend(model.detect(image_batch, verbose=verbose))
        
    return results


def load_model(model_dir, maskflow_config, mode="inference"):
    # Recreate the model in inference mode
    model = FixedMaskRCNN(mode=mode, config=maskflow_config, model_dir=str(model_dir))
    return model
    
    
def load_weights(model, init_with="last"):
    
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    
    elif init_with == "coco":
        
        root_dir = Path(model.model_dir).parent
        coco_model_path = root_dir / "mask_rcnn_coco.h5"
        if not coco_model_path.is_file():
            mrcnn.utils.download_trained_weights(str(coco_model_path))
            
        model.load_weights(str(coco_model_path), by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    elif init_with == "last":
        model_path = model.find_last()
        model.load_weights(str(model_path), by_name=True)

    else:
        log_dir = Path(model.model_dir) / init_with
        model_candidates = sorted(list(log_dir.glob(f"mask_rcnn_*.h5")))
        model_path = model_candidates[-1]
        model.load_weights(str(model_path), by_name=True)
        
        
def export_to_saved_model(model, maksrcnn_model_path):
    # Get keras model and save
    model_keras= model.keras_model

    # All new operations will be in test mode from now on.
    K.set_learning_phase(0)

    # Create output layer with customized names
    prediction_node_names = ["detections", "mrcnn_class", "mrcnn_bbox",
                             "mrcnn_mask", "rois", "rpn_class", "rpn_bbox"]
    prediction_node_names = ["output_" + name for name in prediction_node_names]
    num_output = len(prediction_node_names)

    predidction = []
    for i in range(num_output):
        tf.identity(model_keras.outputs[i], name = prediction_node_names[i])

    sess = K.get_session()

    # Get the object detection graph
    od_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), 
                                                             prediction_node_names)

    with tf.gfile.GFile(str(maksrcnn_model_path), 'wb') as f:
        f.write(od_graph_def.SerializeToString())
        
            
def export_to_dir(model, model_name, final_saved_model_dir, copy_log_dir=False):
    
    if copy_log_dir:
        log_path = Path(model.log_dir).parent / model_name
        shutil.rmtree(log_path, ignore_errors=True)
        shutil.copytree(model.log_dir, log_path)
    else:
        log_path = Path(model.log_dir)
        
    os.makedirs(final_saved_model_dir, exist_ok=True)
    
    config_path = log_path / "config.yml"
    shutil.copy(config_path, final_saved_model_dir)
    
    mrcnn_model_path = final_saved_model_dir / "maskrcnn.pb"
    preprocessing_model_path = final_saved_model_dir / "preprocessing.pb"
    postprocessing_model_path = final_saved_model_dir / "postprocessing.pb"

    export_to_saved_model(model, mrcnn_model_path)
    processing_graph.build_preprocessing_graph(preprocessing_model_path)
    processing_graph.build_postprocessing_graph(postprocessing_model_path)
    
    
def export_to_zip(model, model_name, saved_model_dir, copy_log_dir=False):
    zip_model_path = saved_model_dir / (model_name + ".zip")
    temp_path = tempfile.mkdtemp()
    temp_path = Path(temp_path)
    export_to_dir(model, model_name, temp_path, copy_log_dir=copy_log_dir)
    
    with zipfile.ZipFile(zip_model_path, "w") as z:
        for fpath in Path(temp_path).iterdir():
            z.write(fpath, arcname=fpath.name)
    
    shutil.rmtree(temp_path)
    
    
class FixedMaskRCNN(MaskRCNN):
    """A fixed version of mrcnn.model.MaskRCNN"""
        
    def load_weights(self, filepath, by_name=False, exclude=None):
        super().load_weights(filepath, by_name=by_name, exclude=exclude)
        #self.log_dir = str(Path(filepath).parent)
        
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        
        if self.mode == "inference":
            return
        
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
    
    
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=[]):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gausssian blur with a random sigma in range 0 to 5.
                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]
        
        callbacks += custom_callbacks

        # Train
        mrcnn_model.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        mrcnn_model.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
            verbose=0
        )
        self.epoch = max(self.epoch, epochs)