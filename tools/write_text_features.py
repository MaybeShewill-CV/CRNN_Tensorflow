#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午7:47
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : write_text_features.py
# @IDE: PyCharm Community Edition
"""
Write text features into tensorflow records
"""
import os
import os.path as ops
import argparse
import numpy as np

from data_provider import data_provider
from local_utils import data_utils


def init_args() -> argparse.Namespace:
    """ Parses command line arguments

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Writes text features from train and test data as tensorflow records')
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Path to "Train" and "Test" folders with data')
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='Where to store the generated tfrecords')
    parser.add_argument('-a', '--annotation_file', type=str, default='sample.txt',
                        help='Name of annotations file (in dataset_dir/Train and dataset_dir/Test)')
    parser.add_argument('-v', '--validation_split', type=float, default=0.15,
                        help='Fraction of training data to use for validation. Set to 0 to disable.')
    parser.add_argument('-n', '--normalization', type=str, default=None,
                        help="Perform normalization on images. Can be either 'divide_255' or 'divide_256'")
    return parser.parse_args()


def write_features(dataset_dir: str, save_dir: str, annotation_name: str, validation_split: float, normalization: str):
    """ Processes training and test data creating Tensorflow records.

    :param dataset_dir: root to Train and Test datasets
    :param save_dir: Where to store the tf records
    :param annotation_name: Name of annotations file in each dataset dir
    :param validation_split: Fraction of training data to use for validation
    :param normalization: Perform normalization on images 'divide_255', 'divide_256'
    """
    os.makedirs(save_dir, exist_ok=True)

    print('Initializing the dataset provider... ', end='')

    provider = data_provider.TextDataProvider(dataset_dir=dataset_dir, annotation_name=annotation_name,
                                              validation_set=validation_split > 0, validation_split=validation_split,
                                              shuffle='every_epoch', normalization=normalization.lower())
    print('done.')

    feature_io = data_utils.TextFeatureIO()

    # write train tfrecords
    print('Writing tf records for training...')

    train_images = provider.train.images
    train_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in train_images]
    train_labels = provider.train.labels
    train_imagenames = provider.train.imagenames

    train_tfrecord_path = ops.join(save_dir, 'train_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=train_tfrecord_path, labels=train_labels, images=train_images,
                                     imagenames=train_imagenames)

    # write test tfrecords
    print('Writing tf records for testing...')

    test_images = provider.test.images
    test_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in test_images]
    test_labels = provider.test.labels
    test_imagenames = provider.test.imagenames

    test_tfrecord_path = ops.join(save_dir, 'test_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=test_tfrecord_path, labels=test_labels, images=test_images,
                                     imagenames=test_imagenames)

    # write val tfrecords
    print('Writing tf records for validation...')

    val_images = provider.validation.images
    val_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in val_images]
    val_labels = provider.validation.labels
    val_imagenames = provider.validation.imagenames

    val_tfrecord_path = ops.join(save_dir, 'validation_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=val_tfrecord_path, labels=val_labels, images=val_images,
                                     imagenames=val_imagenames)


if __name__ == '__main__':
    # init args
    args = init_args()
    if not ops.exists(args.dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(args.dataset_dir))

    # write tf records
    write_features(dataset_dir=args.dataset_dir, save_dir=args.save_dir, annotation_name=args.annotation_file,
                   validation_split=args.validation_split, normalization=args.normalization)
