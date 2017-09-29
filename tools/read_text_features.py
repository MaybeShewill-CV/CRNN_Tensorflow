#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 上午11:28
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : read_text_features.py
# @IDE: PyCharm Community Edition
"""
Read text features from tensorflow records
"""
import os
import os.path as ops
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from local_utils import data_utils


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_dir', type=str, help='Where you store the dataset')
    parser.add_argument('--is_vis', type=bool, help='Whether need visualize')

    return parser.parse_args()


def read_features(tfrecords_dir, is_vis=True):
    """

    :param tfrecords_dir:
    :param is_vis:
    :return:
    """
    records_path = [ops.join(tfrecords_dir, tmp) for tmp in os.listdir(tfrecords_dir) if tmp.endswith('tfrecords')]

    feature_io = data_utils.TextFeatureIO()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess_config.gpu_options.allow_growth = False

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        for path in records_path:
            images, labels, imagenames = feature_io.reader.read_features(path, num_epochs=None)
            sh_images, sh_labels, sh_imagenames = tf.train.shuffle_batch(
                tensors=[images, labels, imagenames], batch_size=1, capacity=1000 + 2 * 1, min_after_dequeue=100)
            sh_images, sh_labels, sh_imagenames = sess.run([sh_images, sh_labels, sh_imagenames])
            print(sh_images.shape[0])
            for index, image in enumerate(sh_images):
                print('{:s} label is {:s}'.format(sh_imagenames[index], sh_labels[index]))
                if is_vis:
                    plt.imshow(image[index][:, :, (2, 1, 0)])
                    plt.show()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # read features
    read_features(args.tfrecords_dir)
