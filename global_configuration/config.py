#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午3:25
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : config.py
# @IDE: PyCharm Community Edition
"""
Set some global configuration
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

__C.ARCH = edict()

# Number of units in each LSTM cell
__C.ARCH.HIDDEN_UNITS = 256
# Number of stacked LSTM cells
__C.ARCH.HIDDEN_LAYERS = 2
# Sequence length.  This has to be the width of the final feature map of the CNN, which is input size width / 4
__C.ARCH.SEQ_LENGTH = 25
# Width x height into which training / testing images are resized before feeding into the network
__C.ARCH.INPUT_SIZE = (100, 32)

# Train options
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 40000
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 100
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.1
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.85
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 32
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 32
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 10000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.5
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = False
# Set the test batch size
__C.TEST.BATCH_SIZE = 32
