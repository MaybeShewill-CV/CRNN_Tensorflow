#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-21 下午3:03
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : tf_io_pipline_fast_tools.py
# @IDE: PyCharm
"""
Efficient tfrecords writer interface
"""
import os
import os.path as ops
from multiprocessing import Manager
from multiprocessing import Process
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf
import tqdm

from config import global_config
from local_utils import establish_char_dict

CFG = global_config.cfg

_SAMPLE_INFO_QUEUE = Manager().Queue()
_SENTINEL = ("", [])


def _int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto.
    :param value:
    :return:
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
    :param value:
    :return:
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
    :param value:
    :return:
    """
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _is_valid_jpg_file(image_path):
    """

    :param image_path:
    :return:
    """

    if not ops.exists(image_path):
        return False

    file = open(image_path, 'rb')
    data = file.read(11)
    if data[:4] != '\xff\xd8\xff\xe0' and data[:4] != '\xff\xd8\xff\xe1':
        file.close()
        return False
    if data[6:] != 'JFIF\0' and data[6:] != 'Exif\0':
        file.close()
        return False
    file.close()

    file = open(image_path, 'rb')
    file.seek(-2, 2)
    if file.read() != '\xff\xd9':
        file.close()
        return False

    file.close()

    return True


def _write_tfrecords(tfrecords_writer):
    """

    :param tfrecords_writer:
    :return:
    """
    while True:
        sample_info = _SAMPLE_INFO_QUEUE.get()

        if sample_info == _SENTINEL:
            log.info('Process {:d} finished writing work'.format(os.getpid()))
            tfrecords_writer.close()
            break

        sample_path = sample_info[0]
        sample_label = sample_info[1]

        if _is_valid_jpg_file(sample_path):
            log.error('Image file: {:d} is not a valid jpg file'.format(sample_path))
            continue

        try:
            image = cv2.imread(sample_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            image = image.tostring()
        except IOError as err:
            log.error(err)
            continue

        features = tf.train.Features(feature={
            'labels': _int64_feature(sample_label),
            'images': _bytes_feature(image),
            'imagepaths': _bytes_feature(sample_path)
        })
        tf_example = tf.train.Example(features=features)
        tfrecords_writer.write(tf_example.SerializeToString())
        log.debug('Process: {:d} get sample from sample_info_queue[current_size={:d}], '
                  'and write it to local file at time: {}'.format(
                   os.getpid(), _SAMPLE_INFO_QUEUE.qsize(), time.strftime('%H:%M:%S')))


class _FeatureIO(object):
    """
    Feature IO Base Class
    """
    def __init__(self, char_dict_path, ord_map_dict_path):
        """

        :param char_dict_path:
        :param ord_map_dict_path:
        """
        self._char_dict = establish_char_dict.CharDictBuilder.read_char_dict(char_dict_path)
        self._ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(ord_map_dict_path)
        return

    def char_to_int(self, char):
        """
        convert char into int index, first convert the char into it's ord
        number and the convert the ord number into int index which is stored
        in ord_map_dict.json file
        :param char: single character
        :return: the int index of the character
        """
        str_key = str(ord(char)) + '_ord'
        try:
            result = int(self._ord_map[str_key])
            return result
        except KeyError:
            raise KeyError("Character {} missing in ord_map.json".format(char))

    def int_to_char(self, number):
        """
        convert the int index into char
        :param number: Can be passed as string representing the integer value to look up.
        :return: Character corresponding to 'number' in the char_dict
        """
        # 1 is the default value in sparse_tensor_to_str() This will be skipped when building the resulting strings
        if number == 1 or number == '1':
            return '\x00'
        else:
            return self._char_dict[str(number) + '_ord']

    def encode_labels(self, labels):
        """
        Convert a batch of text labels into int index labels
        :param labels: List of text labels such as ['hello world', 'fuck world', ...]
        :return: Two list. One is a list of int index labels another is
        a list of label length
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
        values = np.array([self._ord_map[str(tmp) + '_index'] for tmp in values])
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

    def sparse_tensor_to_str_for_tf_serving(self, decode_indices, decode_values, decode_dense_shape):
        """

        :param decode_indices:
        :param decode_values:
        :param decode_dense_shape:
        :return:
        """
        indices = decode_indices
        values = decode_values
        # Translate from consecutive numbering into ord() values
        values = np.array([self._ord_map[str(tmp) + '_index'] for tmp in values])
        dense_shape = decode_dense_shape

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


class CrnnFeatureReader(_FeatureIO):
    """
        Implement the crnn feature reader
    """

    def __init__(self, char_dict_path, ord_map_dict_path, flags='train'):
        """

        :param char_dict_path:
        :param ord_map_dict_path:
        :param flags:
        """
        super(CrnnFeatureReader, self).__init__(char_dict_path, ord_map_dict_path)
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

    def inputs(self, tfrecords_path, batch_size, num_threads):
        """

        :param tfrecords_path:
        :param batch_size:
        :param num_threads:
        :return: input_images, input_labels, input_image_names
        """
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


class CrnnFeatureWriter(_FeatureIO):
    """
    crnn tensorflow tfrecords writer
    """

    def __init__(self, annotation_infos, lexicon_infos,
                 char_dict_path, ord_map_dict_path,
                 tfrecords_save_dir, writer_process_nums, dataset_flag):
        """
        Every file path should be checked outside of the class, make sure the file path is valid when you
        call the class. Make sure the info list is not empty when you call the class. I will put all the
        sample information into a queue which may cost lots of memory if you've got really large dataset
        :param annotation_infos: example info list [(image_absolute_path, lexicon_index), ...]
        :param lexicon_infos: lexicon info list [lexicon1, lexicon2, ...]
        :param char_dict_path: char dict file path
        :param ord_map_dict_path: ord map dict file path
        :param tfrecords_save_dir: tfrecords save dir
        :param writer_process_nums: the process nums of which will write the tensorflow examples
        into local tensorflow records file. Each thread will write down examples into its own
        local tensorflow records file
        :param dataset_flag: dataset flag which will be the tfrecords file's prefix name
        """
        super(CrnnFeatureWriter, self).__init__(
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path
        )

        # init sample info queue
        self._dataset_flag = dataset_flag
        self._annotation_infos = annotation_infos
        self._lexicon_infos = lexicon_infos
        self._writer_process_nums = writer_process_nums
        self._init_example_info_queue()
        self._tfrecords_save_dir = tfrecords_save_dir

    def _init_example_info_queue(self):
        """
        Read index file and put example info into SAMPLE_INFO_QUEUE
        :return:
        """
        log.info('Start filling {:s} dataset sample information queue...'.format(self._dataset_flag))

        t_start = time.time()
        for annotation_info in tqdm.tqdm(self._annotation_infos):
            image_path = annotation_info[0]
            lexicon_index = annotation_info[1]

            try:
                lexicon_label = [self._lexicon_infos[lexicon_index]]
                encoded_label, _ = self.encode_labels(lexicon_label)

                _SAMPLE_INFO_QUEUE.put((image_path, encoded_label[0]))
            except IndexError:
                log.error('Lexicon doesn\'t contain lexicon index {:d}'.format(lexicon_index))
                continue
        for i in range(self._writer_process_nums):
            _SAMPLE_INFO_QUEUE.put(_SENTINEL)
        log.debug('Complete filling dataset sample information queue[current size: {:d}], cost time: {:.5f}s'.format(
            _SAMPLE_INFO_QUEUE.qsize(),
            time.time() - t_start
        ))

    def run(self):
        """

        :return:
        """
        log.info('Start write tensorflow records for {:s}...'.format(self._dataset_flag))

        process_pool = []
        tfwriters = []
        for i in range(self._writer_process_nums):
            tfrecords_save_name = '{:s}_{:d}.tfrecords'.format(self._dataset_flag, i + 1)
            tfrecords_save_path = ops.join(self._tfrecords_save_dir, tfrecords_save_name)

            tfrecords_io_writer = tf.python_io.TFRecordWriter(path=tfrecords_save_path)
            process = Process(
                target=_write_tfrecords,
                name='Subprocess_{:d}'.format(i + 1),
                args=(tfrecords_io_writer,)
            )
            process_pool.append(process)
            tfwriters.append(tfrecords_io_writer)
            process.start()

        for process in process_pool:
            process.join()

        log.info('Finished writing down the tensorflow records file')
