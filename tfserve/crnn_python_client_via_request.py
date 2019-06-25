#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : eldon
# @Site    : https://github.com/eldon/CRNN_Tensorflow
# @File    : crnn_python_client_via_request.py
"""
Use shadow net to recognize the scene text of a single image
"""
import json
import os.path as ops

import cv2
import glog as logger
import numpy as np
import requests
import wordninja

from config import global_config
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg
SERVER_URL = 'http://localhost:8501/v1/models/crnn:predict'
CHAR_DICT_PATH = './data/char_dict/char_dict_en.json'
ORD_MAP_DICT_PATH = './data/char_dict/ord_map_en.json'


def request_crnn_predict(image_path):
    """
    request crnn predict
    :param image_path:
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # constrain image input size to (100, 32)
    image = cv2.resize(image, tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
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

    preds = codec.sparse_tensor_to_str_for_tf_serving(
        decode_indices=outputs['decodes_indices'],
        decode_values=outputs['decodes_values'],
        decode_dense_shape=outputs['decodes_dense_shape'],
    )[0]
    preds = ' '.join(wordninja.split(preds))

    logger.info('Predict image {:s} result: {:s}'.format(
        ops.split(image_path)[1], preds)
    )


if __name__ == '__main__':
    """
    test code
    """
    import sys

    img_path = sys.argv[1]
    print(img_path)
    request_crnn_predict(sys.argv[1])
