#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : eldon
# @Site    : https://github.com/eldon/CRNN_Tensorflow
# @File    : test_shadownet.py
"""
Use shadow net to recognize the scene text of a single image
"""
import argparse
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glog as logger
import json
import wordninja

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

import requests

CFG = global_config.cfg
SERVER_URL = 'http://localhost:8501/v1/models/crnn:predict'
CHAR_DICT_PATH = './data/char_dict/char_dict_en.json'
ORD_MAP_DICT_PATH = './data/char_dict/ord_map_en.json'

def request_crnn_predict(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # constrain image input size to (100, 32)
    image = cv2.resize(image, CFG.ARCH.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    image = np.array(image, np.float32) / 127.5 - 1.0

    response = requests.post(
        SERVER_URL,
        data=json.dumps({
            'inputs': [image.tolist()],  # has to be in column format; not a fixed output size
        }),
    )
    response.raise_for_status()
    outputs = response.json()['outputs']

    # this part can likely be optimized, but oh well
    codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=CHAR_DICT_PATH,
        ord_map_dict_path=ORD_MAP_DICT_PATH,
    )

    preds = codec.unpack_sparse_tensor_to_str(
        outputs['decodes_indices'],
        outputs['decodes_values'],
        outputs['decodes_dense_shape'],
    )[0]
    preds = ' '.join(wordninja.split(preds))

    logger.info('Predict image {:s} result: {:s}'.format(
        ops.split(image_path)[1], preds)
    )

if __name__ == '__main__':
    import sys
    img_path = sys.argv[1]
    print(img_path)
    request_crnn_predict(sys.argv[1])

