#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-14 下午3:18
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : export_saved_model.py
# @IDE: PyCharm
"""
Convert ckpt model into tensorflow saved model
"""
import os.path as ops
import argparse
import glog as log

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import saved_model as sm

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--export_dir', type=str, help='The model export dir')
    parser.add_argument('-i', '--ckpt_path', type=str, help='The pretrained ckpt model weights file path')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')

    return parser.parse_args()


def build_saved_model(ckpt_path, export_dir):
    """
    Convert source ckpt weights file into tensorflow saved model
    :param ckpt_path:
    :param export_dir:
    :return:
    """

    if ops.exists(export_dir):
        raise ValueError('Export dir must be a dir path that does not exist')

    assert ops.exists(ops.split(ckpt_path)[0])

    # build inference tensorflow graph
    image_size = tuple(CFG.ARCH.INPUT_SIZE)
    image_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, image_size[1], image_size[0], 3],
        name='input_tensor')

    # set crnn net
    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    # compute inference logits
    inference_ret = net.inference(
        inputdata=image_tensor,
        name='shadow_net',
        reuse=False
    )

    # beam search decode
    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=CFG.ARCH.SEQ_LENGTH * np.ones(1),
        merge_repeated=False
    )

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=ckpt_path)

        # set model save builder
        saved_builder = sm.builder.SavedModelBuilder(export_dir)

        # add tensor need to be saved
        saved_input_tensor = sm.utils.build_tensor_info(image_tensor)
        saved_prediction_tensor = sm.utils.build_tensor_info(decodes[0])

        # build SignatureDef protobuf
        signatur_def = sm.signature_def_utils.build_signature_def(
            inputs={'input_tensor': saved_input_tensor},
            outputs={'prediction': saved_prediction_tensor},
            method_name=sm.signature_constants.PREDICT_METHOD_NAME
        )

        # add graph into MetaGraphDef protobuf
        saved_builder.add_meta_graph_and_variables(
            sess,
            tags=[sm.tag_constants.SERVING],
            signature_def_map={sm.signature_constants.PREDICT_OUTPUTS: signatur_def}
        )

        # save model
        saved_builder.save()

    return


def test_load_saved_model(saved_model_dir, char_dict_path, ord_map_dict_path):
    """

    :param saved_model_dir:
    :param char_dict_path:
    :param ord_map_dict_path:
    :return:
    """
    image = cv2.imread('data/test_images/test_01.jpg', cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(
        src=image,
        dsize=tuple(CFG.ARCH.INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )
    image = np.array(image, np.float32) / 127.5 - 1.0
    image = np.expand_dims(image, 0)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        meta_graphdef = sm.loader.load(
            sess,
            tags=[sm.tag_constants.SERVING],
            export_dir=saved_model_dir)

        signature_def_d = meta_graphdef.signature_def
        signature_def_d = signature_def_d[sm.signature_constants.PREDICT_OUTPUTS]

        image_input_tensor = signature_def_d.inputs['input_tensor']
        prediction_tensor = signature_def_d.outputs['prediction']

        input_tensor = sm.utils.get_tensor_from_tensor_info(image_input_tensor, sess.graph)
        predictions = sm.utils.get_tensor_from_tensor_info(prediction_tensor, sess.graph)

        prediction_val = sess.run(predictions, feed_dict={input_tensor: image})

        codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path
        )

        prediction_val = codec.sparse_tensor_to_str(prediction_val)[0]

        log.info('Predict image result ----> {:s}'.format(prediction_val))

        plt.figure('CRNN Model Demo')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.show()


if __name__ == '__main__':
    """
    build saved model
    """
    # init args
    args = init_args()

    # build saved model
    build_saved_model(args.ckpt_path, args.export_dir)

    # test build saved model
    test_load_saved_model(args.export_dir, args.char_dict_path, args.ord_map_dict_path)
