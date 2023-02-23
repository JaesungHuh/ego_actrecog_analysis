#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

_C.BN.FREEZE = False

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# finetune
_C.TRAIN.FINETUNE = False

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.ENABLE = True

# Dataset for testing.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.TEST.CHECKPOINT_TYPE = "pytorch"

_C.TEST.SLIDE = CfgNode()
_C.TEST.SLIDE.ENABLE = False
_C.TEST.SLIDE.WIN_SIZE = 1.
_C.TEST.SLIDE.HOP_SIZE = 1.
_C.TEST.SLIDE.LABEL_FRAME = 0.5
_C.TEST.SLIDE.INSIDE_ACTION_BOUNDS = 'strict'

# -----------------------------------------------------------------------------
# FEATURE EXTRACTION OPTIONS - only for omnivore
# -----------------------------------------------------------------------------
_C.TEST.FEATURE_EXTRACTION = True

_C.TEST.OUTPUT_MEAT = True

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.ARCH = "slowfast"

# Model name
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.NUM_CLASSES = [400, ]

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slowonly"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.MODEL.FC_INIT_STD = 0.01

###############
# TSM configs #
###############
_C.MODEL.MODALITY = 'RGB' # RGB, Flow, Both
_C.MODEL.BASE_MODEL = 'resnet50'
_C.MODEL.NUM_SEGMENTS = 8
_C.MODEL.SEGMENT_LENGTH = [1,5] # RGB:1, Flow: 5
_C.MODEL.BEFORE_SOFTMAX = True
_C.MODEL.CROP_NUM = 1
_C.MODEL.CONCENSUS_TYPE = 'avg'
_C.MODEL.PARTIAL_BN = True
_C.MODEL.PRETRAINED = 'kinetics'
_C.MODEL.IS_SHIFT = True
_C.MODEL.SHIFT_DIV = 8
_C.MODEL.SHIFT_PLACE = 'blockres'
_C.MODEL.FC_LR5 = False
_C.MODEL.TEMPORAL_POOL = False
_C.MODEL.NON_LOCAL = False

# -----------------------------------------------------------------------------
# Slowfast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

_C.DATA.USE_REPEATED_AUG = False

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The number of frames of the input clip.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# List of input frame channel dimensions.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA.TEST_CROP_SIZE = 256

_C.DATA.FRAME_SAMPLING = 'like slowfast'

_C.DATA.USE_RAND_AUGMENT = False

_C.DATA.MULTI_LABEL = False

_C.DATA.USE_REPEATED_AUG = False

# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# Use Mixed Precision Training
_C.SOLVER.USE_MIXED_PRECISION = False

# If > 0.0, use label smoothing
_C.SOLVER.SMOOTHING = 0.0

# Clip Grad
_C.SOLVER.CLIP_GRAD = None


_C.WANDB = CfgNode()

_C.WANDB.ENABLE = True

_C.WANDB.RUN_ID = ""

# ---------------------------------------------------------------------------- #
# Mixup options
# ---------------------------------------------------------------------------- #

_C.MIXUP = CfgNode()

# mixup alpha, mixup enabled if > 0. (default: 0.8)
_C.MIXUP.MIXUP_ALPHA = 0.0

# cutmix alpha, cutmix enabled if > 0. (default: 1.0)
_C.MIXUP.CUTMIX_ALPHA = 0.0

# cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
_C.MIXUP.CUTMIX_MINMAX = None

# Probability of performing mixup or cutmix when either/both is enabled
_C.MIXUP.MIXUP_PROB = 0.5

# Probability of switching to cutmix when both mixup and cutmix enabled
_C.MIXUP.MIXUP_SWITCH_PROB = 0.5

# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.MIXUP.MIXUP_MODE = "batch"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NUM_GPUS = 1 

# Number of machine to use for the job.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.NUM_SHARDS = 1

# The index of the current machine.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.SHARD_ID = 0

# Output basedir.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# -----------------------------------------------------------------------------
# EPIC-KITCHENS Dataset options
# -----------------------------------------------------------------------------
_C.EPICKITCHENS = CfgNode()

_C.EPICKITCHENS.VISUAL_DATA_DIR = ""

_C.EPICKITCHENS.VIDEO_DURS = "EPIC_100_video_info.csv"

#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.EPICKITCHENS.ANNOTATIONS_DIR = ""

_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

#############
# ✓✓✓✓✓✓✓✓✓ #
#############
_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TEST_SPLIT = "validation"

_C.EPICKITCHENS.TRAIN_PLUS_VAL = False




def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
