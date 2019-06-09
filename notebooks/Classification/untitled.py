
flags.DEFINE_float(
    'batch_norm_momentum',
    default=None,
    help=('Batch normalization layer momentum of moving average to override.'))
flags.DEFINE_float(
    'batch_norm_epsilon',
    default=None,
    help=('Batch normalization layer epsilon to override..'))

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_bool(
    'use_bfloat16',
    default=False,
    help=('Whether to use bfloat16 as activation for training.'))

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))
flags.DEFINE_bool(
    'export_to_tpu', default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))

flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'moving_average_decay', default=0.9999,
    help=('Moving average decay rate.'))

flags.DEFINE_float(
    'weight_decay', default=1e-5,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate', default=None,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_float(
    'drop_connect_rate', default=None,
    help=('Drop connect rate for the network.'))