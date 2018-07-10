import os
import re
import sys
from pathlib import Path
import datetime
import logging
import multiprocessing
import tempfile
import shutil
import zipfile

import numpy as np
import tqdm
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from tensorflow.python.framework import graph_util

from mrcnn.model import norm_boxes_graph
from mrcnn.model import denorm_boxes_graph
from mrcnn.model import resnet_graph
from mrcnn.model import compute_backbone_shapes
from mrcnn.model import build_rpn_model
from mrcnn.model import ProposalLayer
from mrcnn.model import parse_image_meta_graph
from mrcnn.model import DetectionTargetLayer
from mrcnn.model import fpn_classifier_graph
from mrcnn.model import build_fpn_mask_graph
from mrcnn.model import rpn_class_loss_graph
from mrcnn.model import rpn_bbox_loss_graph
from mrcnn.model import mrcnn_class_loss_graph
from mrcnn.model import mrcnn_bbox_loss_graph
from mrcnn.model import mrcnn_mask_loss_graph
from mrcnn.model import DetectionLayer
from mrcnn.model import mold_image
from mrcnn.model import unmold_image
from mrcnn.model import compose_image_meta

from mrcnn.utils import generate_pyramid_anchors
from mrcnn.utils import norm_boxes
from mrcnn.utils import denorm_boxes
from mrcnn.utils import unmold_mask
from mrcnn.utils import resize_image
from mrcnn.utils import download_trained_weights

from .config import save_parameters
from . import processing_graph
from .callbacks import TrainValTensorBoard
from .data_generator import DataGenerator


def load_model(model_dir, config, training_name=None, mode="inference", init_with="coco"):
    return Maskflow(model_dir=model_dir, config=config, training_name=training_name, mode=mode, init_with=init_with)


def export_to_saved_model(model, maksrcnn_model_path):

    assert model.mode == "inference"

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

    params_path = log_path / "parameters.yml"
    shutil.copy(params_path, final_saved_model_dir)

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


class Maskflow:

    def __init__(self, model_dir, config, training_name=None, mode="inference", init_with="coco"):
        """
        mode: Either "training" or "inference".
        config: A Sub-class of the mrcnn Config class.
        model_dir: Directory to save training logs and trained weights.
        last_training: the name of a sub-directory of model_dir. If set last training will be resumed.
        init_with: "coco", "imagenet" or None
        """

        assert mode in ['training', 'inference']        
        
        self.log = logging.getLogger("Maskflow")
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                                      "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

        self.model_dir = Path(model_dir)
        self.config = config
        self.mode = mode
        self.training_name = training_name
        self.init_with = init_with

        if mode == "inference" and not self.training_name:
            raise Exception("In inference mode you need to provide a training name "
                            "which is a subddicrectory of model_dir.")

        self._set_log_dir()
        
        self.keras_model = self._build_keras_model(self.mode, self.config)

    def _set_log_dir(self):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        if self.training_name:
            self.log_dir = self.model_dir / self.training_name
            self.log.info(f"Set log_dir to {self.log_dir}")
        else:
            # Directory for training logs
            self.log_dir = self.model_dir / f"{self.config.NAME.lower()}{now:%Y%m%dT%H%M}"
            self.log.info(f"Set new training log_dir: {self.log_dir}")

        # Continue from we left of. Get epoch and date from the file name
        # A sample model path might look like:
        # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
        pattern = r"mask\_rcnn\_[\w-]+(\d{4})"
        def get_epoch(fname):
            m = re.match(pattern, fname)
            if m:
                return int(m.group(1))

        fnames = [fname.stem for fname in self.log_dir.glob("*.h5")]
        epochs = [get_epoch(fname) for fname in fnames]
        epochs = list(filter(lambda x: x is not None, epochs))
        if len(epochs) > 0:
            self.epoch = max(epochs) + 1
            self.log.info(f"Found weights from epoch {self.epoch - 1}")

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = self.log_dir / f"mask_rcnn_{self.config.NAME.lower()}_{{epoch:04d}}.h5"
        
        Path(self.log_dir.parent).mkdir(parents=True, exist_ok=True)

    def _build_keras_model(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        self.log.info("Start building Keras model.")

        K.clear_session()

        # Fix types
        config.DETECTION_MAX_INSTANCES = int(config.DETECTION_MAX_INSTANCES)
        
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        with tf.name_scope('Inputs'):
            # Inputs
            input_image = KL.Input(
                shape=[None, None, 3], name="input_image")
            input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                        name="input_image_meta")
            if mode == "training":
                # RPN GT
                input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
                input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

                # Detection GT (class IDs, bounding boxes, and masks)
                # 1. GT Class IDs (zero padded)
                input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
                
                # 2. GT Boxes in pixels (zero padded)
                # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
                input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
                
                # Normalize coordinates
                gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
                
                # 3. GT Masks (zero padded)
                # [batch, height, width, MAX_GT_INSTANCES]
                if config.USE_MINI_MASK:
                    input_gt_masks = KL.Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                                              name="input_gt_masks", dtype=bool)
                else:
                    input_gt_masks = KL.Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                                              name="input_gt_masks", dtype=bool)
            elif mode == "inference":
                # Anchors in normalized coordinates
                input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        with tf.name_scope('Resnet_Graph'):
            # Build the shared convolutional layers.
            # Bottom-up Layers
            # Returns a list of the last layers of each stage, 5 in total.
            # Don't create the thead (stage 5), so we pick the 4th item in the list.
            if callable(config.BACKBONE):
                _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                    train_bn=config.TRAIN_BN)
            else:
                _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                                 stage5=True, train_bn=config.TRAIN_BN)

            with tf.name_scope('Top_Down_Layers'):
                # Top-down Layers
                # TODO: add assert to varify feature map sizes match what's in config
                P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
                P4 = KL.Add(name="fpn_p4add")([
                    KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
                P3 = KL.Add(name="fpn_p3add")([
                    KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
                P2 = KL.Add(name="fpn_p2add")([
                    KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
                # Attach 3x3 conv to all P layers to get the final feature maps.
                P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
                P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
                P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
                P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
                # P6 is used for the 5th anchor scale in RPN. Generated by
                # subsampling from P5 with stride of 2.
                P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

                # Note that P6 is used in RPN, but not in the classifier heads.
                rpn_feature_maps = [P2, P3, P4, P5, P6]
                mrcnn_feature_maps = [P2, P3, P4, P5]

        with tf.name_scope('Anchors'):
            # Anchors
            if mode == "training":
                anchors = self._get_anchors(config.IMAGE_SHAPE)
                # Duplicate across the batch dimension because Keras requires it
                # TODO: can this be optimized to avoid duplicating the anchors?
                anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
                # A hack to get around Keras's bad support for constants
                anchors = KL.Lambda(lambda x: K.variable(anchors), name="anchors")(input_image)
            else:
                anchors = input_anchors

        with tf.name_scope('RPN_Graph'):
            # RPN Model
            rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                  len(config.RPN_ANCHOR_RATIOS),
                                  config.TOP_DOWN_PYRAMID_SIZE)

            # Loop through pyramid layers
            layer_outputs = []  # list of lists
            for p in rpn_feature_maps:
                layer_outputs.append(rpn([p]))
            # Concatenate layer outputs
            # Convert from list of lists of level outputs to list of lists
            # of outputs across levels.
            # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
            output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
            outputs = list(zip(*layer_outputs))
            outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                       for o, n in zip(outputs, output_names)]

            rpn_class_logits, rpn_class, rpn_bbox = outputs

        with tf.name_scope('Proposal_Graph'):
            # Generate proposals
            # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
            # and zero padded.
            proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
                else config.POST_NMS_ROIS_INFERENCE
            rpn_rois = ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=config.RPN_NMS_THRESHOLD,
                name="ROI",
                config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            with tf.name_scope('Detection_Target_Layer'):
                # Generate detection targets
                # Subsamples proposals and generates target outputs for training
                # Note that proposal class IDs, gt_boxes, and gt_masks are zero
                # padded. Equally, returned rois and targets are zero padded.
                rois, target_class_ids, target_bbox, target_mask =\
                    DetectionTargetLayer(config, name="proposal_targets")([
                        target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

                # Network Heads
                # TODO: verify that this handles zero padded ROIs
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                    fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                         config.POOL_SIZE, config.NUM_CLASSES,
                                         train_bn=config.TRAIN_BN,
                                         fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

                # TODO: clean up (use tf.identify if necessary)
                output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            with tf.name_scope('Mask_Detection_Layer'):
                mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)

            # TODO: I don't think it can work easily with Keras because we use `.fit_generator()`.
            # See https://github.com/keras-team/keras/blob/b6bb0be10ad84d456bb1cea535a4a922d7f2e2c6/keras/callbacks.py#L865
            # And https://github.com/keras-team/keras/issues/10472
            #with tf.name_scope('Training_Summaries'):
            #    self.summary_tensors = [input_image]
            #    tf.summary.image('Input_Image', tf.cast(input_image, tf.float32), max_outputs=10)
            #    true_rois_image = tf.image.draw_bounding_boxes(input_image, input_gt_boxes)
            #    tf.summary.image('True_ROIs', tf.cast(true_rois_image, tf.float32), max_outputs=10)
            #    predicted_rois_image = tf.image.draw_bounding_boxes(input_image, output_rois)
            #    tf.summary.image('Predicted_ROIs', tf.cast(predicted_rois_image, tf.float32), max_outputs=10)

            with tf.name_scope('Losses'):
                # Losses
                rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                    [input_rpn_match, rpn_class_logits])
                rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                    [input_rpn_bbox, input_rpn_match, rpn_bbox])
                class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                    [target_class_ids, mrcnn_class_logits, active_class_ids])
                bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                    [target_bbox, target_class_ids, mrcnn_bbox])
                mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                    [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            with tf.name_scope('Detection_Target_Layer'):
                # Detections
                # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
                # normalized coordinates
                detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            with tf.name_scope('Mask_Detection_Layer'):
                # Create masks for detections
                detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
                mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from .parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
            
        self.log.info("Keras model built.")
        return model

    def _get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def load_weights(self):
        """Load weights if needed with "coco", "imagenet" or
        last training weights.
        """
        if self.epoch > 0:
            weight_candidates = sorted(list(self.log_dir.glob(f"mask_rcnn_*.h5")))
            weight_path = weight_candidates[-1]
            self.log.info(f"Load weights from previous training {weight_path}.")
            self._load_weights(str(weight_path), by_name=True)

        elif self.init_with == "imagenet":
            raise Exception("TODO")

        elif self.init_with == "coco":
            self.log.info("Load weights with coco pretrained model.")
            root_dir = Path(self.model_dir).parent
            coco_model_path = root_dir / "mask_rcnn_coco.h5"
            if not coco_model_path.is_file():
                download_trained_weights(str(coco_model_path))

            self._load_weights(str(coco_model_path), by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])

    def _load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import saving
        #from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        self.log.info("Done loading weights.")

    def _set_trainable(self, layer_regex, keras_model=None, indent=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        self.log.info("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        if hasattr(keras_model, "inner_model"):
            layers = keras_model.inner_model.layers
        else:
            layers = keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                self.log.info(f"In model: {layer.name}")
                self._set_trainable(layer_regex, keras_model=layer, indent=indent + 1)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable:
                indent_str = "\t" * indent
                self.log.info(f"{indent_str}{layer.name:20} ({layer.__class__.__name__})")

    def _compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss",  "rpn_bbox_loss",
                      "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))      
            
        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)
            
    def train(self, train_dataset, val_dataset, epochs, layers,
              augmentation=None, custom_callbacks=[], learning_rate=None):
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

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Copy params file to log directory
        save_parameters(self.config.params, self.log_dir / "parameters.yml")

        if not learning_rate:
            learning_rate = self.config.LEARNING_RATE

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
        train_generator = DataGenerator(train_dataset, self.config, shuffle=True,
                                        augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True,
                                      batch_size=self.config.BATCH_SIZE)

        # Train
        self.log.info(f"Starting at epoch {self.epoch}. Learning Rate={learning_rate}")
        self.log.info(f"Checkpoint Path: {self.checkpoint_path}")
        self._set_trainable(layers)
        
        # It will set self.train_model
        self._compile(learning_rate, self.config.LEARNING_MOMENTUM)
        
        # Callbacks
        tb = TrainValTensorBoard(log_dir=str(self.log_dir), histogram_freq=0, write_graph=True, write_images=False)
        mc = keras.callbacks.ModelCheckpoint(str(self.checkpoint_path), verbose=0, save_weights_only=True)
        callbacks = [tb, mc]
        callbacks += custom_callbacks

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()
            
        self.keras_model.fit_generator(train_generator,
                                       initial_epoch=self.epoch,
                                       epochs=epochs,
                                       steps_per_epoch=self.config.STEPS_PER_EPOCH,
                                       callbacks=callbacks,
                                       validation_data=val_generator,
                                       validation_steps=self.config.VALIDATION_STEPS,
                                       max_queue_size=50,
                                       workers=workers,
                                       use_multiprocessing=True)
        
        self.epoch = max(self.epoch, epochs)

    def _split_as_batches(self, images, config):

        images = np.array(images)

        image_batches = []
        config.BATCH_SIZE = config.IMAGES_PER_GPU
        n_batches = max(1, images.shape[0] // config.BATCH_SIZE + 0)

        for i in range(n_batches):
            n = i * config.BATCH_SIZE
            image_batch = images[n:n + config.BATCH_SIZE]
            image_batches.append(image_batch)

        return image_batches

    def _detect(self, images):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self._mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self._get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self._unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def _unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                           image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.
        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def _mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def predict(self, image, progress=True):
        if image.ndim == 3 and image.shape[-1] <= 3:
            image = np.array([image])

        # Split image in multiple batches
        image_batches = self._split_as_batches(image, self.config)

        # Run Prediction on Batches
        results = []
        for image_batch in tqdm.tqdm(image_batches, disable=not progress):
            results.extend(self._detect(image_batch))
        return results

