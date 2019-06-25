#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-27 下午4:15
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV
# @File    : crnn_python_client_via_grpc.py.py
# @IDE: PyCharm
"""
test python tensorflow client
"""
import time
from argparse import ArgumentParser

import grpc
import numpy as np
import cv2
from tensorflow import saved_model as sm
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from config import global_config
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg

CODEC = tf_io_pipline_fast_tools.CrnnFeatureReader(
    char_dict_path='./data/char_dict/char_dict_en.json',
    ord_map_dict_path='./data/char_dict/ord_map_en.json'
)


def parse_args():
    """

    :return:
    """
    parser = ArgumentParser(description='Request a TensorFlow server for a prediction on the image')
    parser.add_argument('-s', '--server',
                        dest='server',
                        default='localhost:9000',
                        help='prediction service host:port')
    parser.add_argument("-i", "--image",
                        dest="image",
                        default='',
                        help="path to image in JPEG format", )
    parser.add_argument('-p', '--image_path',
                        dest='image_path',
                        default='./data/test_images/test_01.jpg',
                        help='path to images folder', )
    parser.add_argument('-b', '--batch_mode',
                        dest='batch_mode',
                        default='true',
                        help='send image as batch or one-by-one')
    args = parser.parse_args()

    return args.server, args.image, args.image_path, args.batch_mode == 'true'


def make_request(image_path, server):
    """

    :param image_path:
    :param server:
    :return:
    """
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (CFG.ARCH.INPUT_SIZE[0], CFG.ARCH.INPUT_SIZE[1]), interpolation=cv2.INTER_LINEAR)
    image = np.array(image, np.float32) / 127.5 - 1.0

    image_list = np.array([image], dtype=np.float32)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'crnn'
    request.model_spec.signature_name = sm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    request.inputs['input_tensor'].CopyFrom(make_tensor_proto(
        image_list, shape=[1, CFG.ARCH.INPUT_SIZE[1], CFG.ARCH.INPUT_SIZE[0], 3]))

    try:
        result = stub.Predict(request, 10.0)

        return result
    except Exception as err:
        print(err)
        return None


def convert_predict_response_into_nparray(response, output_tensor_name):
    """

    :param response:
    :param output_tensor_name:
    :return:
    """
    dims = response.outputs[output_tensor_name].tensor_shape.dim
    shape = tuple(d.size for d in dims)

    return np.reshape(response.outputs[output_tensor_name].int64_val, shape)


def post_process(tf_serving_request_result):
    """

    :param tf_serving_request_result:
    :return:
    """
    decode_indices = convert_predict_response_into_nparray(
        tf_serving_request_result,
        'decodes_indices'
    )
    decode_values = convert_predict_response_into_nparray(
        tf_serving_request_result,
        'decodes_values'
    )
    decode_dense_shape = convert_predict_response_into_nparray(
        tf_serving_request_result,
        'decodes_dense_shape'
    )

    prediction = CODEC.sparse_tensor_to_str_for_tf_serving(
        decode_indices=decode_indices,
        decode_values=decode_values,
        decode_dense_shape=decode_dense_shape
    )[0]

    print('Prediction: {:s}'.format(prediction))

    return


def main():
    """

    :return:
    """
    server, image, image_path, batch_mode = parse_args()

    print('Server: {}'.format(server))

    start = time.time()

    request_result = make_request(image_path, server)

    post_process(request_result)

    end = time.time()
    time_diff = end - start
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    """
    test code
    """
    main()
