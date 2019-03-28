#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午3:25
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : global_config.py
# @IDE: PyCharm Community Edition
"""
Set some global configuration
"""
import logging

from easydict import EasyDict as edict
from local_utils import log_utils

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

__C.ARCH = edict()

# Number of units in each LSTM cell
__C.ARCH.HIDDEN_UNITS = 256
# Number of stacked LSTM cells
__C.ARCH.HIDDEN_LAYERS = 2
# Sequence length.  This has to be the width of the final feature map of the CNN, which is input size width / 4
# __C.ARCH.SEQ_LENGTH = 70  # cn dataset
__C.ARCH.SEQ_LENGTH = 25  # synth90k dataset
# Width x height into which training / testing images are resized before feeding into the network
# __C.ARCH.INPUT_SIZE = (280, 32)  # cn dataset
__C.ARCH.INPUT_SIZE = (100, 32)  # synth90k dataset
# Number of channels in images
__C.ARCH.INPUT_CHANNELS = 3
# Number character classes
# __C.ARCH.NUM_CLASSES = 5825  # cn dataset
__C.ARCH.NUM_CLASSES = 37  # synth90k dataset

# Train options
__C.TRAIN = edict()

# Use early stopping?
__C.TRAIN.EARLY_STOPPING = False
# Wait at least this many epochs without improvement in the cost function
__C.TRAIN.PATIENCE_EPOCHS = 6
# Expect at least this improvement in one epoch in order to reset the early stopping counter
__C.TRAIN.PATIENCE_DELTA = 1e-3

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 2000000
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 100
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.01
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.9
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 32
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 32
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 500000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1
# Update learning rate in jumps?
__C.TRAIN.LR_STAIRCASE = True
# Set multi process nums
__C.TRAIN.CPU_MULTI_PROCESS_NUMS = 6
# Set Gpu nums
__C.TRAIN.GPU_NUM = 2
# Set moving average decay
__C.TRAIN.MOVING_AVERAGE_DECAY = 0.9999
# Set val display step
__C.TRAIN.VAL_DISPLAY_STEP = 1000

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.6
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = False
# Set the test batch size
__C.TEST.BATCH_SIZE = 32
