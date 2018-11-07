#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 下午3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Test shadow net script
"""
import importlib
import os
import os.path as ops
import sys
from typing import Tuple

import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math

from local_utils import data_utils
from local_utils.config_utils import load_config
from local_utils.log_utils import compute_accuracy
from crnn_model import crnn_model
from easydict import EasyDict


def init_args() -> Tuple[argparse.Namespace, EasyDict]:
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-c', '--chardict_dir', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-w', '--weights_path', type=str, required=True,
                        help='Path to pre-trained weights')
    parser.add_argument('-n', '--num_classes', type=int, required=True,
                        help='Force number of character classes to this number. '
                             'Use 37 to run with the demo data. '
                             'Set to 0 for auto (read from files in charset_dir)')
    parser.add_argument('-f', '--config_file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-v', '--visualize', type=bool, default=False,
                        help='Whether to display images')
    parser.add_argument('-b', '--one_batch', default=False,
                        action='store_true', help='Test only one batch of the dataset')
    parser.add_argument('-j', '--num_threads', type=int,
                        default=int(os.cpu_count() / 2),
                        help='Number of threads to use in batch shuffling')

    args = parser.parse_args()
    config = load_config(args.config_file)
    if args.dataset_dir:
        config.cfg.PATH.TFRECORDS_DIR = args.dataset_dir
    if args.chardict_dir:
        config.cfg.PATH.CHAR_DICT_DIR = args.chardict_dir

    return args, config.cfg


def test_shadownet(weights_path: str, cfg: EasyDict, visualize: bool, process_all_data: bool=True, num_threads: int=4,
                   num_classes: int=0):
    """

    :param tfrecords_dir: Directory with test_feature.tfrecords
    :param charset_dir: Path to char_dict.json and ord_map.json (generated with write_text_features.py)
    :param weights_path: Path to stored weights
    :param cfg: configuration EasyDict (e.g. global_config.config.cfg)
    :param visualize: whether to display the images
    :param process_all_data:
    :param num_threads: Number of threads for tf.train.(shuffle_)batch
    :param num_classes: Number of different characters in the dataset
    """
    decoder = data_utils.TextFeatureIO(char_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'char_dict.json'),
                                       ord_map_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'ord_map.json')).reader
    input_images, input_labels, input_image_names = decoder.read_features(cfg, cfg.TEST.BATCH_SIZE, num_threads, False)

    num_classes = len(decoder.char_dict) + 1 if num_classes == 0 else num_classes
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=cfg.ARCH.HIDDEN_UNITS,
                               layers_nums=cfg.ARCH.HIDDEN_LAYERS,
                               num_classes=num_classes)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=input_images)

    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, cfg.ARCH.SEQ_LENGTH * np.ones(cfg.TEST.BATCH_SIZE),
                                               merge_repeated=False)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    test_sample_count = sum(1 for _ in tf.python_io.tf_record_iterator(
        ops.join(cfg.PATH.TFRECORDS_DIR, 'test_feature.tfrecords')))
    num_iterations = int(math.ceil(test_sample_count / cfg.TEST.BATCH_SIZE)) if process_all_data \
        else 1

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        print('Start predicting...')

        accuracy = 0
        for epoch in range(num_iterations):
            predictions, images, labels, image_names = sess.run([decoded, input_images, input_labels, input_image_names])
            image_names = np.reshape(image_names, newshape=image_names.shape[0])
            image_names = [tmp.decode('utf-8') for tmp in image_names]

            labels = decoder.sparse_tensor_to_str(labels)
            predictions = decoder.sparse_tensor_to_str(predictions[0])

            accuracy += compute_accuracy(labels, predictions, display=False)

            for index, image in enumerate(images):
                print('Predict {:s} image with gt label: {:s} **** predicted label: {:s}'.format(
                    image_names[index], labels[index], predictions[index]))
                # avoid accidentally displaying for the whole dataset
                if visualize and not process_all_data:
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.show()

        # We compute a mean of means, so we need the sample sizes to be constant
        # (BATCH_SIZE) for this to equal the actual mean
        accuracy /= num_iterations
        print('Mean test accuracy is {:5f}'.format(accuracy))


if __name__ == '__main__':
    args, cfg = init_args()

    test_shadownet(weights_path=args.weights_path, cfg=cfg, process_all_data=not args.one_batch,
                   visualize=args.visualize, num_threads=args.num_threads, num_classes=args.num_classes)
