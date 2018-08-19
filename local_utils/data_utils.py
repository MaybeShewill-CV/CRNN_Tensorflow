#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午6:46
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : data_utils.py
# @IDE: PyCharm Community Edition
"""
Implement some utils used to convert image and it's corresponding label into tfrecords
"""
from typing import List

import numpy as np
import tensorflow as tf
import os
import os.path as ops
import sys

from local_utils import establish_char_dict


class FeatureIO(object):
    """
        Implement the base writer class
    """
    def __init__(self, char_dict_path=ops.join(os.getcwd(), 'data/char_dict/char_dict.json'),
                 ord_map_dict_path=ops.join(os.getcwd(), 'data/char_dict/ord_map.json')):
        self.__char_dict = establish_char_dict.CharDictBuilder.read_char_dict(char_dict_path)
        self.__ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(ord_map_dict_path)
        return

    @property
    def char_dict(self):
        """

        :return:
        """
        return self.__char_dict

    @staticmethod
    def int64_feature(value):
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
        if is_int is False:
            value = value_tmp
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
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

    @staticmethod
    def bytes_feature(value):
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

    def char_to_int(self, char: str) -> int:
        """

        :param char:
        :return:
        """
        temp = ord(char)
        # convert upper character into lower character
        if 65 <= temp <= 90:
            temp = temp + 32

        for k, v in self.__ord_map.items():
            if v == str(temp):
                temp = int(k)
                return temp
        raise KeyError("Character {} missing in ord_map.json".format(char))

        # TODO
        # Here implement a double way dict or two dict to quickly map ord and it's corresponding index

    def int_to_char(self, number: int) -> str:
        """ Return the character corresponding to the given integer.

        :param number: Can be passed as string representing the integer value to look up.
        :return: Character corresponding to 'number' in the char_dict
        """
        if number == '1':
            return '*'
        if number == 1:
            return '*'
        else:
            return self.__char_dict[str(number)]

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

    def sparse_tensor_to_str(self, sparse_tensor: tf.SparseTensor) -> List[str]:
        """
        :param sparse_tensor: prediction or ground truth label
        :return: String value of the sparse tensor
        """
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        values = np.array([self.__ord_map[str(tmp)] for tmp in values])
        dense_shape = sparse_tensor.dense_shape

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]
        for number_list in number_lists:
            str_lists.append([self.int_to_char(val) for val in number_list])
        for str_list in str_lists:
            res.append(''.join(c for c in str_list if c != '*'))
        return res


class TextFeatureWriter(FeatureIO):
    """
        Implement the crnn feature writer
    """
    def __init__(self):
        super(TextFeatureWriter, self).__init__()
        return

    def write_features(self, tfrecords_path, labels, images, imagenames):
        """

        :param tfrecords_path:
        :param labels:
        :param images:
        :param imagenames:
        :return:
        """
        assert len(labels) == len(images) == len(imagenames)

        labels, length = self.encode_labels(labels)

        if not ops.exists(ops.split(tfrecords_path)[0]):
            os.makedirs(ops.split(tfrecords_path)[0])

        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for index, image in enumerate(images):
                features = tf.train.Features(feature={
                    'labels': self.int64_feature(labels[index]),
                    'images': self.bytes_feature(image),
                    'imagenames': self.bytes_feature(imagenames[index])
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(index+1, len(images), imagenames[index]))
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        return


class TextFeatureReader(FeatureIO):
    """
        Implement the crnn feature reader
    """
    def __init__(self):
        super(TextFeatureReader, self).__init__()
        return

    @staticmethod
    def read_features(tfrecords_path, num_epochs):
        """

        :param tfrecords_path:
        :param num_epochs:
        :return:
        """
        assert ops.exists(tfrecords_path)

        filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'images': tf.FixedLenFeature((), tf.string),
                                               'imagenames': tf.FixedLenFeature([1], tf.string),
                                               'labels': tf.VarLenFeature(tf.int64),
                                           })
        image = tf.decode_raw(features['images'], tf.uint8)
        images = tf.reshape(image, [32, 100, 3])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        imagenames = features['imagenames']
        return images, labels, imagenames


class TextFeatureIO(object):
    """
        Implement a crnn feture io manager
    """
    def __init__(self):
        """

        """
        self.__writer = TextFeatureWriter()
        self.__reader = TextFeatureReader()
        return

    @property
    def writer(self):
        """

        :return:
        """
        return self.__writer

    @property
    def reader(self):
        """

        :return:
        """
        return self.__reader
