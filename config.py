import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# ----------------------------------------------------------------------------------------------------------------------
# Data settings
# ----------------------------------------------------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = ''
# Dataset mean/std type
_C.DATA.MEAN_AND_STD_TYPE = "default"

# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'

# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = False
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4
# Data image filename format
_C.DATA.FNAME_FORMAT = '{}.jpeg'
# Data debug, when debug is True, only use few images
_C.DATA.DEBUG = False

# ----------------------------------------------------------------------------------------------------------------------
# Model settings
# ----------------------------------------------------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = ''
# Model name
_C.MODEL.NAME = ''
# Pretrained weight from checkpoint, could be overwritten by command line argument
_C.MODEL.PRETRAINED = True
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''

# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Simple-ViT Model
# _C.MODEL.SIMPLE_VIT = CN()
# _C.MODEL.SIMPLE_VIT.IN_CHANS = 3
# _C.MODEL.SIMPLE_VIT.PATCH_SIZE = 32
# _C.MODEL.SIMPLE_VIT.DIM = 1024
# _C.MODEL.SIMPLE_VIT.DEPTH = 6
# _C.MODEL.SIMPLE_VIT.HEADS = 16
# _C.MODEL.SIMPLE_VIT.MLP_DIM = 2048
# _C.MODEL.SIMPLE_VIT.DIM_HEAD = 64

# ----------------------------------------------------------------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.START_EPOCH = 0

_C.TRAIN.EPOCHS = 300

_C.TRAIN.WARMUP_EPOCHS = 10

_C.TRAIN.WEIGHT_DECAY = 0.05

_C.TRAIN.BASE_LR = 5e-4

_C.TRAIN.WARMUP_LR = 5e-7

_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps, could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory, could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# train learning rate decay
_C.TRAIN.LAYER_LR_DECAY = 1.0
# batch norm is in evaluation mode when training
_C.TRAIN.EVAL_BN_WHEN_TRAINING = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()

_C.TRAIN.LR_SCHEDULER.NAME = ''
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()

_C.TRAIN.OPTIMIZER.NAME = ''
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# ----------------------------------------------------------------------------------------------------------------------
# Augmentation settings
# ----------------------------------------------------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 0.2
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# ----------------------------------------------------------------------------------------------------------------------
# Testing settings
# ----------------------------------------------------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# ----------------------------------------------------------------------------------------------------------------------
# Quantization setting
# ----------------------------------------------------------------------------------------------------------------------
_C.QUANT = CN()

_C.QUANT.METRIC = ""

_C.QUANT.CALIB_SIZE = 32

_C.QUANT.N_BITS = 8

_C.QUANT.SIGN_BIT = 1

_C.QUANT.EXPONENT = 4

_C.QUANT.PER_CHANNEL = True

_C.QUANT.EM_MX = True

_C.QUANT.BIAS_MX = True

_C.QUANT.SPECIFIC_CFG_PATH = ''

_C.FP32 = False

_C.DATAFLOW_VAL = False

# ----------------------------------------------------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------------------------------------------------

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False

_C.ONLY_CPU = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.disable_amp or args.only_cpu:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args=None):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if args is not None:
        update_config(config, args)

    return config
