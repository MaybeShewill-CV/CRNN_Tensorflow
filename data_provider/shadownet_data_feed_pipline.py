#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-26 下午9:03
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : shadownet_data_feed_pipline.py
# @IDE: PyCharm
"""
nsfw数据feed pipline
"""
import os
import os.path as ops
import random

import glob
import glog as log
import tensorflow as tf

from config import global_config
from local_utils import establish_char_dict
from data_provider import tf_io_pipline_tools

CFG = global_config.cfg


class CrnnDataProducer(object):
    """
    Convert raw image file into tfrecords
    """

    def __init__(self, dataset_dir, char_dict_path=None, ord_map_dict_path=None):
        """
        init crnn data producer
        :param dataset_dir: image dataset root dir
        :param char_dict_path: char dict path
        :param ord_map_dict_path: ord map dict path
        """
        if not ops.exists(dataset_dir):
            raise ValueError('Dataset dir {:s} not exist'.format(dataset_dir))

        # Check image source data
        self._image_dir = ops.join(dataset_dir, 'images')
        self._train_annotation_file_path = ops.join(dataset_dir, 'annotation_train.txt')
        self._test_annotation_file_path = ops.join(dataset_dir, 'annotation_test.txt')
        self._val_annotation_file_path = ops.join(dataset_dir, 'annotation_val.txt')
        self._lexicon_file_path = ops.join(dataset_dir, 'lexicon.txt')
        self._char_dict_path = char_dict_path
        self._ord_map_dict_path = ord_map_dict_path

        if not self._is_source_data_complete():
            raise ValueError('Source image data is not complete, '
                             'please check if one of the image folder '
                             'or index file is not exist')

        # Init training example information
        log.info('Start initialzie example information.....')
        self._lexicon_list = []
        self._train_example_paths = []
        self._train_example_labels = []
        self._test_example_paths = []
        self._test_example_labels = []
        self._val_example_paths = []
        self._val_example_labels = []
        self._init_dataset_example_info()

        # Check if need generate char dict map
        if char_dict_path is None or ord_map_dict_path is None:
            os.makedirs('./data/char_dict', exist_ok=True)
            self._char_dict_path = ops.join('./data/char_dict', 'char_dict.json')
            self._ord_map_dict_path = ops.join('./data/char_dict', 'ord_map.json')
        else:
            self._char_dict_path = char_dict_path
            self._ord_map_dict_path = ord_map_dict_path
        self._generate_char_dict()

        # Init tfrecords writer
        self._tfrecords_io_writer = tf_io_pipline_tools.TextFeatureIO(
            char_dict_path=self._char_dict_path, ord_map_dict_path=self._ord_map_dict_path).writer

    def generate_tfrecords(self, save_dir, step_size=10000):
        """
        Generate tensorflow records file
        :param save_dir:
        :param step_size: generate a tfrecord every step_size examples
        :return:
        """

        def _split_writing_tfrecords_task(_example_paths, _example_labels, _flags='train'):

            _split_example_paths = []
            _split_example_labels = []
            _split_tfrecords_save_paths = []

            _example_nums = len(_example_paths)

            for i in range(0, _example_nums, step_size):
                _split_example_paths.append(_example_paths[i:i + step_size])
                _split_example_labels.append(_example_labels[i:i + step_size])

                if i + step_size > _example_nums:
                    _split_tfrecords_save_paths.append(
                        ops.join(save_dir, '{:s}_{:d}_{:d}.tfrecords'.format(_flags, i, _example_nums)))
                else:
                    _split_tfrecords_save_paths.append(
                        ops.join(save_dir, '{:s}_{:d}_{:d}.tfrecords'.format(_flags, i, i + step_size)))

            return _split_example_paths, _split_example_labels, _split_tfrecords_save_paths

        # make save dirs
        os.makedirs(save_dir, exist_ok=True)

        # generate training example tfrecords
        log.info('Generating training example tfrecords')

        train_example_paths_split, train_example_labels_split, train_tfrecords_save_paths = \
            _split_writing_tfrecords_task(
                self._train_example_paths, self._train_example_labels, _flags='train'
            )

        for index, example_paths in enumerate(train_example_paths_split):
            self._tfrecords_io_writer.write_features(
                example_image_paths=example_paths,
                example_image_labels=train_example_labels_split[index],
                tfrecords_save_path=train_tfrecords_save_paths[index]
            )

        log.info('Generate training example tfrecords complete')

        # generate val example tfrecords
        log.info('Generating validation example tfrecords')

        val_example_paths_split, val_example_labels_split, val_tfrecords_save_paths = \
            _split_writing_tfrecords_task(
                self._val_example_paths, self._val_example_labels, _flags='val'
            )

        for index, example_paths in enumerate(val_example_paths_split):
            self._tfrecords_io_writer.write_features(
                example_image_paths=example_paths,
                example_image_labels=val_example_labels_split[index],
                tfrecords_save_path=val_tfrecords_save_paths[index]
            )

        log.info('Generate validation example tfrecords complete')

        # generate test example tfrecords
        log.info('Generating testing example tfrecords')

        test_example_paths_split, test_example_labels_split, test_tfrecords_save_paths = \
            _split_writing_tfrecords_task(
                self._test_example_paths, self._test_example_labels, _flags='test'
            )

        for index, example_paths in enumerate(test_example_paths_split):
            self._tfrecords_io_writer.write_features(
                example_image_paths=example_paths,
                example_image_labels=test_example_labels_split[index],
                tfrecords_save_path=test_tfrecords_save_paths[index]
            )

        log.info('Generate testing example tfrecords complete')

        return

    def _is_source_data_complete(self):
        """
        Check if source data complete
        :return:
        """
        return \
            ops.exists(self._train_annotation_file_path) and ops.exists(self._val_annotation_file_path) \
            and ops.exists(self._test_annotation_file_path) and ops.exists(self._lexicon_file_path)

    def _init_dataset_example_info(self):
        """
        organize dataset example information
        :return:
        """
        # establish lexicon list
        with open(self._lexicon_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self._lexicon_list.append(line.rstrip('\r').rstrip('\n'))

        # establish train example info [(image_path1, label1), (image_path2, label2), ...]
        with open(self._train_annotation_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                image_name, image_label = line.rstrip('\r').rstrip('\n').split(' ')
                image_path = ops.join(self._image_dir, image_name)
                self._train_example_paths.append(image_path)
                self._train_example_labels.append(image_label)

        # establish val example info [(image_path1, label1), (image_path2, label2), ...]
        with open(self._val_annotation_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                image_name, image_label = line.rstrip('\r').rstrip('\n').split(' ')
                image_path = ops.join(self._image_dir, image_name)
                self._val_example_paths.append(image_path)
                self._val_example_labels.append(image_label)

        # establish test example info [(image_path1, label1), (image_path2, label2), ...]
        with open(self._test_annotation_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                image_name, image_label = line.rstrip('\r').rstrip('\n').split(' ')
                image_path = ops.join(self._image_dir, image_name)
                self._test_example_paths.append(image_path)
                self._test_example_labels.append(image_label)

    def _generate_char_dict(self):
        """

        :return:
        """
        char_lexicon_set = set()
        for lexcion in self._lexicon_list:
            char_list = [s for s in lexcion]
            char_lexicon_set = char_lexicon_set.union(set(char_list))

        log.info('Char set length: {:d}'.format(len(char_lexicon_set)))

        char_lexicon_list = list(char_lexicon_set)
        char_dict_builder = establish_char_dict.CharDictBuilder()
        char_dict_builder.write_char_dict(char_lexicon_list, save_path=self._char_dict_path)
        char_dict_builder.map_ord_to_index(char_lexicon_list, save_path=self._ord_map_dict_path)

        log.info('Write char dict map complete')


class CrnnDataFeeder(object):
    """
    Read training examples from tfrecords for crnn model
    """

    def __init__(self, dataset_dir, char_dict_path, ord_map_dict_path, flags='train'):
        """

        :param dataset_dir:
        :param char_dict_path:
        :param ord_map_dict_path:
        :param flags:
        """
        self._dataset_dir = dataset_dir

        self._tfrecords_dir = ops.join(dataset_dir, 'tfrecords')
        if not ops.exists(self._tfrecords_dir):
            raise ValueError('{:s} not exist, please check again'.format(self._tfrecords_dir))

        self._dataset_flags = flags.lower()
        if self._dataset_flags not in ['train', 'test', 'val']:
            raise ValueError('flags of the data feeder should be \'train\', \'test\', \'val\'')

        self._char_dict_path = char_dict_path
        self._ord_map_dict_path = ord_map_dict_path
        self._tfrecords_io_reader = tf_io_pipline_tools.TextFeatureIO(
            char_dict_path=self._char_dict_path, ord_map_dict_path=self._ord_map_dict_path).reader
        self._tfrecords_io_reader.dataset_flags = self._dataset_flags

    def sample_counts(self):
        """

        :return:
        """
        tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(self._tfrecords_dir, self._dataset_flags))
        counts = 0

        for record in tfrecords_file_paths:
            counts += sum(1 for _ in tf.python_io.tf_record_iterator(record))

        return counts

    def inputs(self, batch_size, num_epochs):
        """
        dataset feed pipline input
        :param batch_size:
        :param num_epochs:
        :return: A tuple (images, labels), where:
                    * images is a float tensor with shape [batch_size, H, W, C]
                      in the range [-0.5, 0.5].
                    * labels is an int32 tensor with shape [batch_size] with the true label,
                      a number in the range [0, CLASS_NUMS).
        """
        if not num_epochs:
            num_epochs = None

        tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(self._tfrecords_dir, self._dataset_flags))
        random.shuffle(tfrecords_file_paths)

        return self._tfrecords_io_reader.inputs(
            tfrecords_path=tfrecords_file_paths,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_threads=CFG.TRAIN.CPU_MULTI_PROCESS_NUMS
        )


if __name__ == '__main__':
    """
    test code
    """
    # test crnn data producer
    producer = CrnnDataProducer(
        dataset_dir='/media/baidu/Data/Sequence_Recognition/Synth_90K/90kDICT32px',
    )

    producer.generate_tfrecords(
        save_dir='/media/baidu/Data/Sequence_Recognition/Synth_90K/tfrecords',
        step_size=100000
    )
