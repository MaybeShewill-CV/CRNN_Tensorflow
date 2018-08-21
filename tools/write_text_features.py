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
from functools import reduce
import numpy as np

from data_provider import data_provider
from data_provider.data_provider import TextDataset
from local_utils.data_utils import TextFeatureIO
from local_utils.establish_char_dict import CharDictBuilder


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
                        help='Name of annotations file (in dataset_dir/Train and dataset_dir/Test). '
                             'The encoding is assumed to be utf-8')
    parser.add_argument('-v', '--validation_split', type=float, default=0.15,
                        help='Fraction of training data to use for validation. Set to 0 to disable.')
    parser.add_argument('-n', '--normalization', type=str, default=None,
                        help="Perform normalization on images. Can be either 'divide_255' or 'divide_256'")
    parser.add_argument('-c', '--char_maps', type=str, default=None,
                        help='Set the path where character maps will be saved from labels in training and test sets.')
    return parser.parse_args()


def write_tfrecords(dataset: TextDataset, name: str, save_dir: str, char_maps_dir: str=None):
    """

    :param dataset:
    :param name:
    :param save_dir:
    :param char_maps_dir:
    :return:
    """
    tfrecord_path = ops.join(save_dir, '%s_features.tfrecords' % name)
    print('Writing tf records for %s at %s...' % (name, tfrecord_path))

    images = dataset.images
    h, w = dataset.images.shape
    images = [bytes(list(np.reshape(tmp, [w * h * 3]))) for tmp in images]
    labels = dataset.labels
    imagenames = dataset.imagenames

    if char_maps_dir is not None:
        os.makedirs(os.path.dirname(char_maps_dir), exist_ok=True)
        # FIXME: rereading every time is a bit silly...
        try:
            d = CharDictBuilder.read_char_dict(os.path.join(char_maps_dir, "char_dict.json"))
            all_chars = set(map(lambda k: chr(int(k)), d.keys()))
        except FileNotFoundError:
            all_chars = set()
        all_chars = all_chars.union(reduce(lambda a, b: set(a).union(set(b)), labels))
        CharDictBuilder.write_char_dict(all_chars, os.path.join(char_maps_dir, "char_dict.json"))
        CharDictBuilder.map_ord_to_index(all_chars, os.path.join(char_maps_dir, "ord_map.json"))
        print("  (character maps written)")

        char_dict_path=os.path.join(char_maps_dir, "char_dict.json")
        ord_map_dict_path=os.path.join(char_maps_dir, "ord_map.json")
    else:
        char_dict_path = os.path.join("data/char_dict", "char_dict.json")
        ord_map_dict_path = os.path.join("data/char_dict", "ord_map.json")

    feature_io = TextFeatureIO(char_dict_path, ord_map_dict_path)
    feature_io.writer.write_features(tfrecords_path=tfrecord_path, labels=labels, images=images,
                                     imagenames=imagenames)


def write_features(dataset_dir: str, save_dir: str, annotation_name: str, validation_split: float, normalization: str,
                   char_maps: str):
    """ Processes training and test data creating Tensorflow records.

    :param dataset_dir: root to Train and Test datasets
    :param save_dir: Where to store the tf records
    :param annotation_name: Name of annotations file in each dataset dir
    :param validation_split: Fraction of training data to use for validation
    :param normalization: Perform normalization on images 'divide_255', 'divide_256'
    :param build_char_maps: Whether to extract character maps from training and test labels
    """
    os.makedirs(save_dir, exist_ok=True)

    print('Initializing the dataset provider... ', end='', flush=True)

    provider = data_provider.TextDataProvider(dataset_dir=dataset_dir, annotation_name=annotation_name,
                                              validation_set=validation_split > 0, validation_split=validation_split,
                                              shuffle='every_epoch', normalization=normalization)
    print('done.')

    write_tfrecords(provider.train, "training", save_dir, char_maps)

    write_tfrecords(provider.test, "test", save_dir, char_maps)

    write_tfrecords(provider.validation, "validation", save_dir, char_maps)


if __name__ == '__main__':
    # init args
    args = init_args()
    if not ops.exists(args.dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(args.dataset_dir))

    # write tf records
    write_features(dataset_dir=args.dataset_dir, save_dir=args.save_dir, annotation_name=args.annotation_file,
                   validation_split=args.validation_split, normalization=args.normalization,
                   char_maps=args.char_maps)
