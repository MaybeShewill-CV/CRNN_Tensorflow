#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-26 下午9:00
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : tf_io_pipline_tools.py
# @IDE: PyCharm
"""
Implement some utils used to convert image and it's corresponding label into tfrecords
"""
import os
import os.path as ops
import sys

import numpy as np
import cv2
import tensorflow as tf
import glog as log

from local_utils import establish_char_dict
from config import global_config

CFG = global_config.cfg


def _int64_feature(value):
    """
        Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if not is_int:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """
        Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_float = True
    for val in value:
        if not isinstance(val, int):
            is_float = False
            value_tmp.append(float(val))
    if is_float is False:
        value = value_tmp
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """
        Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class _FeatureIO(object):
    """
        Implement the base writer class
    """

    def __init__(self, char_dict_path, ord_map_dict_path):
        self._char_dict = establish_char_dict.CharDictBuilder.read_char_dict(char_dict_path)
        self._ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(ord_map_dict_path)
        return

    @property
    def char_dict(self):
        """

        :return:
        """
        return self._char_dict

    @staticmethod
    def int64_feature(value):
        """
            Wrapper for inserting int64 features into Example proto.
        """
        return _int64_feature(value)

    @staticmethod
    def float_feature(value):
        """
            Wrapper for inserting float features into Example proto.
        """
        return _float_feature(value)

    @staticmethod
    def bytes_feature(value):
        """
            Wrapper for inserting bytes features into Example proto.
        """
        return _bytes_feature(value)

    def char_to_int(self, char):
        """

        :param char:
        :return:
        """
        temp = ord(char)
        for k, v in self._ord_map.items():
            if v == str(temp):
                return int(k)
        raise KeyError("Character {} missing in ord_map.json".format(char))

    def int_to_char(self, number):
        """ Return the character corresponding to the given integer.

        :param number: Can be passed as string representing the integer value to look up.
        :return: Character corresponding to 'number' in the char_dict
        """
        # 1 is the default value in sparse_tensor_to_str() This will be skipped when building the resulting strings
        if number == 1 or number == '1':
            return '\x00'
        else:
            return self._char_dict[str(number)]

    def encode_labels(self, labels):
        """
            encode the labels for ctc loss
        :param labels:
        :return:
        """
        encoded_labels = []
        lengths = []
        for label in labels:
            encode_label = [self.char_to_int(char) for char in label]
            encoded_labels.append(encode_label)
            lengths.append(len(label))
        return encoded_labels, lengths

    def sparse_tensor_to_str(self, sparse_tensor):
        """
        :param sparse_tensor: prediction or ground truth label
        :return: String value of the sparse tensor
        """
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        # Translate from consecutive numbering into ord() values
        values = np.array([self._ord_map[str(tmp)] for tmp in values])
        dense_shape = sparse_tensor.dense_shape

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]
        for number_list in number_lists:
            # Translate from ord() values into characters
            str_lists.append([self.int_to_char(val) for val in number_list])
        for str_list in str_lists:
            # int_to_char() returns '\x00' for an input == 1, which is the default
            # value in number_lists, so we skip it when building the result
            res.append(''.join(c for c in str_list if c != '\x00'))
        return res


class _TextFeatureWriter(_FeatureIO):
    """
        Implement the crnn feature writer
    """

    def __init__(self, char_dict_path, ord_map_dict_path):
        super(_TextFeatureWriter, self).__init__(char_dict_path, ord_map_dict_path)
        return

    def write_features(self, example_image_paths, example_image_labels, tfrecords_save_path):
        """

        :param example_image_paths:
        :param example_image_labels:
        :param tfrecords_save_path:
        :return:
        """

        example_image_labels, example_image_labels_length = self.encode_labels(example_image_labels)

        with tf.python_io.TFRecordWriter(tfrecords_save_path) as writer:
            for index, image_path in enumerate(example_image_paths):

                with open(image_path, 'rb') as f:
                    check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    log.error('Image file {:s} is not complete'.format(image_path))
                    continue

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
                image = image.tostring()

                features = tf.train.Features(feature={
                    'labels': _int64_feature(example_image_labels[index]),
                    'images': _bytes_feature(image),
                    'imagepaths': _bytes_feature(image_path)
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(
                    index + 1, len(example_image_paths), ops.basename(image_path))
                )
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        return


class _TextFeatureReader(_FeatureIO):
    """
        Implement the crnn feature reader
    """

    def __init__(self, char_dict_path, ord_map_dict_path, flags='train'):
        """

        :param char_dict_path:
        :param ord_map_dict_path:
        :param flags:
        """
        super(_TextFeatureReader, self).__init__(char_dict_path, ord_map_dict_path)
        self._dataset_flag = flags.lower()
        return

    @property
    def dataset_flags(self):
        """

        :return:
        """
        return self._dataset_flag

    @dataset_flags.setter
    def dataset_flags(self, value):
        """

        :value:
        :return:
        """
        if not isinstance(value, str):
           raise ValueError('Dataset flags shoule be str')

        if value.lower() not in ['train', 'val', 'test']:
            raise ValueError('Dataset flags shoule be within \'train\', \'val\', \'test\'')

        self._dataset_flag = value

    @staticmethod
    def _augment_for_train(input_images, input_labels, input_image_paths):
        """

        :param input_images:
        :param input_labels:
        :param input_image_paths:
        :return:
        """
        return input_images, input_labels, input_image_paths

    @staticmethod
    def _augment_for_validation(input_images, input_labels, input_image_paths):
        """

        :param input_images:
        :param input_labels:
        :param input_image_paths:
        :return:
        """
        return input_images, input_labels, input_image_paths

    @staticmethod
    def _normalize(input_images, input_labels, input_image_paths):
        """

        :param input_images:
        :param input_labels:
        :param input_image_paths:
        :return:
        """
        input_images = tf.subtract(tf.divide(input_images, 127.5), 1.0)
        return input_images, input_labels, input_image_paths

    @staticmethod
    def _extract_features_batch(serialized_batch):
        """

        :param serialized_batch:
        :return:
        """
        features = tf.parse_example(
            serialized_batch,
            features={'images': tf.FixedLenFeature([], tf.string),
                      'imagepaths': tf.FixedLenFeature([], tf.string),
                      'labels': tf.VarLenFeature(tf.int64),
                      }
        )
        bs = features['images'].shape[0]
        images = tf.decode_raw(features['images'], tf.uint8)
        w, h = tuple(CFG.ARCH.INPUT_SIZE)
        images = tf.cast(x=images, dtype=tf.float32)
        images = tf.reshape(images, [bs, h, w, CFG.ARCH.INPUT_CHANNELS])

        labels = features['labels']
        labels = tf.cast(labels, tf.int32)

        imagepaths = features['imagepaths']

        return images, labels, imagepaths

    def inputs(self, tfrecords_path, batch_size, num_epochs, num_threads):
        """

        :param tfrecords_path:
        :param batch_size:
        :param num_epochs:
        :param num_threads:
        :return: input_images, input_labels, input_image_names
        """

        if not num_epochs:
            num_epochs = None

        dataset = tf.data.TFRecordDataset(tfrecords_path)

        dataset = dataset.batch(batch_size, drop_remainder=True)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(map_func=self._extract_features_batch,
                              num_parallel_calls=num_threads)
        if self._dataset_flag == 'train':
            dataset = dataset.map(map_func=self._augment_for_train,
                                  num_parallel_calls=num_threads)
        else:
            dataset = dataset.map(map_func=self._augment_for_validation,
                                  num_parallel_calls=num_threads)
        dataset = dataset.map(map_func=self._normalize,
                              num_parallel_calls=num_threads)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        if self._dataset_flag != 'test':
            dataset = dataset.shuffle(buffer_size=1000)
            # repeat num epochs
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next(name='{:s}_IteratorGetNext'.format(self._dataset_flag))


class TextFeatureIO(object):
    """
        Implement a crnn feature io manager
    """

    def __init__(self, char_dict_path, ord_map_dict_path):
        """
        """
        self._writer = _TextFeatureWriter(char_dict_path, ord_map_dict_path)
        self._reader = _TextFeatureReader(char_dict_path, ord_map_dict_path)
        return

    @property
    def writer(self):
        """

        :return:
        """
        return self._writer

    @property
    def reader(self):
        """

        :return:
        """
        return self._reader
