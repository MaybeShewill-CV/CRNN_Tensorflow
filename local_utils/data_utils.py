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
import os
import os.path as ops
import sys

import numpy as np
import tensorflow as tf

from local_utils import establish_char_dict


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

    def char_to_int(self, char):
        """

        :param char:
        :return:
        """
        temp = ord(char)
        for k, v in self._ord_map.items():
            if v == str(temp):
                temp = int(k)
                return temp
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
                sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(
                    index+1, len(images), imagenames[index])
                )
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        return


class _TextFeatureReader(_FeatureIO):
    """
        Implement the crnn feature reader
    """
    def __init__(self, char_dict_path, ord_map_dict_path):
        super(_TextFeatureReader, self).__init__(char_dict_path, ord_map_dict_path)
        return

    def read_features(self, cfg, batch_size, num_threads):
        """

        :param cfg:
        :param batch_size:
        :param num_threads:
        :return: input_images, input_labels, input_image_names
        """

        tfrecords_path = os.path.join(cfg.PATH.TFRECORDS_DIR, "train_feature.tfrecords")
        assert ops.exists(tfrecords_path), "tfrecords file not found: %s" % tfrecords_path

        def extract_batch(x):
            return self._extract_features_batch(x, cfg.ARCH.INPUT_SIZE, cfg.ARCH.INPUT_CHANNELS)

        dataset = tf.data.TFRecordDataset(tfrecords_path)
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE, drop_remainder=True)
        dataset = dataset.map(extract_batch, num_parallel_calls=num_threads)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(batch_size * num_threads))
        dataset = dataset.prefetch(buffer_size=batch_size * num_threads)
        iterator = dataset.make_one_shot_iterator()
        input_images, input_labels, input_image_names = iterator.get_next()
        return input_images, input_labels, input_image_names

    @staticmethod
    def _extract_features_batch(serialized_batch, input_size, input_channels):
        """

        :param serialized_batch:
        :param input_size:
        :param input_channels:
        :return:
        """
        features = tf.parse_example(serialized_batch,
                                    features={'images': tf.FixedLenFeature((), tf.string),
                                              'imagenames': tf.FixedLenFeature([1], tf.string),
                                              'labels': tf.VarLenFeature(tf.int64), })
        bs = features['images'].shape[0]
        images = tf.decode_raw(features['images'], tf.uint8)
        w, h = input_size
        images = tf.cast(x=images, dtype=tf.float32)
        images = tf.reshape(images, [bs, h, w, input_channels])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        imagenames = features['imagenames']
        return images, labels, imagenames


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
