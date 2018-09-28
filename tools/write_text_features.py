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
from local_utils.config_utils import load_config
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
    parser.add_argument('-c', '--charset_dir', type=str, default=None,
                        help='Path were character maps extracted from the labels in the dataset will be saved')
    parser.add_argument('-f', '--config_file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-a', '--annotation_file', type=str, default='sample.txt',
                        help='Name of annotations file (in dataset_dir/Train and dataset_dir/Test). '
                             'The encoding is assumed to be utf-8')
    parser.add_argument('-v', '--validation_split', type=float, default=0.15,
                        help='Fraction of training data to use for validation. Set to 0 to disable.')
    parser.add_argument('-n', '--normalization', type=str, default=None,
                        help="Perform normalization on images. Can be either 'divide_255' or 'divide_256'")
    return parser.parse_args()


def write_tfrecords(dataset: TextDataset, name: str, save_dir: str, charset_dir: str=None):
    """

    :param dataset:
    :param name: Name of the dataset (e.g. "train", "test", or "validation")
    :param save_dir: Where to store the tf records
    :param charset_dir: If not None, extract character maps from labels and merge with any char_dict already present
    """
    tfrecord_path = ops.join(save_dir, '%s_feature.tfrecords' % name)
    print('Writing tf records for %s at %s...' % (name, tfrecord_path))

    images = dataset.images
    h, w, c = images.shape[1:]  # shape is num samples x height x width x num channels
    images = [bytes(list(np.reshape(tmp, [w * h * c]))) for tmp in images]
    labels = dataset.labels
    imagenames = dataset.imagenames

    if dataset.num_examples < 1:
        print("ERROR: No samples or labels or any found. Is your dataset ok?")
        return

    if charset_dir is not None:
        os.makedirs(os.path.dirname(charset_dir), exist_ok=True)
        # FIXME: rereading every time is a bit silly...
        try:
            d = CharDictBuilder.read_char_dict(os.path.join(charset_dir, "char_dict.json"))
            all_chars = set(map(lambda k: chr(int(k)), d.keys()))
        except FileNotFoundError:
            all_chars = set()
        all_chars = all_chars.union(reduce(lambda a, b: set(a).union(set(b)), labels))
        CharDictBuilder.write_char_dict(all_chars, os.path.join(charset_dir, "char_dict.json"))
        CharDictBuilder.map_ord_to_index(all_chars, os.path.join(charset_dir, "ord_map.json"))
        print("  (character maps written)")

        char_dict_path=os.path.join(charset_dir, "char_dict.json")
        ord_map_dict_path=os.path.join(charset_dir, "ord_map.json")
    else:
        char_dict_path = os.path.join("data/char_dict", "char_dict.json")
        ord_map_dict_path = os.path.join("data/char_dict", "ord_map.json")

    feature_io = TextFeatureIO(char_dict_path, ord_map_dict_path)
    feature_io.writer.write_features(tfrecords_path=tfrecord_path, labels=labels, images=images,
                                     imagenames=imagenames)


if __name__ == '__main__':
    args = init_args()

    config = load_config(args.config_file)

    if not ops.exists(args.dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(args.dataset_dir))

    os.makedirs(args.save_dir, exist_ok=True)

    print('Initializing the dataset provider...')

    provider = data_provider.TextDataProvider(dataset_dir=args.dataset_dir, annotation_name=args.annotation_file,
                                              validation_set=args.validation_split > 0,
                                              validation_split=args.validation_split, shuffle='every_epoch',
                                              normalization=args.normalization, input_size=config.cfg.ARCH.INPUT_SIZE)

    write_tfrecords(provider.train, "train", args.save_dir, args.charset_dir)
    write_tfrecords(provider.test, "test", args.save_dir, args.charset_dir)
    write_tfrecords(provider.validation, "val", args.save_dir, args.charset_dir)
