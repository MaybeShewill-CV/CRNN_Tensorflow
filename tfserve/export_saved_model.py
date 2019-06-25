#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-14 下午3:18
# @Author  : MaybeShewill-CV, eldon
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : export_saved_model.py
# @IDE: PyCharm
"""
Convert ckpt model into tensorflow saved model
"""
import argparse
import os.path as ops

import numpy as np
import tensorflow as tf
from tensorflow import saved_model as sm

from config import global_config
from crnn_model import crnn_net

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
        indices_output_tensor_info = sm.utils.build_tensor_info(decodes[0].indices)
        values_output_tensor_info = sm.utils.build_tensor_info(decodes[0].values)
        dense_shape_output_tensor_info = sm.utils.build_tensor_info(decodes[0].dense_shape)

        # build SignatureDef protobuf
        signatur_def = sm.signature_def_utils.build_signature_def(
            inputs={'input_tensor': saved_input_tensor},
            outputs={
                'decodes_indices': indices_output_tensor_info,
                'decodes_values': values_output_tensor_info,
                'decodes_dense_shape': dense_shape_output_tensor_info,
            },
            method_name=sm.signature_constants.PREDICT_METHOD_NAME,
        )

        # add graph into MetaGraphDef protobuf
        saved_builder.add_meta_graph_and_variables(
            sess,
            tags=[sm.tag_constants.SERVING],
            signature_def_map={sm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signatur_def},
        )

        # save model
        saved_builder.save()

    return


if __name__ == '__main__':
    """
    build saved model
    """
    # init args
    args = init_args()

    # build saved model
    build_saved_model(args.ckpt_path, args.export_dir)
