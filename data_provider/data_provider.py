#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : data_provider.py
# @IDE: PyCharm Community Edition
"""
Provide the training and testing data for shadow net
"""
import os.path as ops
from typing import Tuple, Union

import numpy as np
import copy
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from global_configuration import config
from data_provider import base_data_provider


class TextDataset(base_data_provider.Dataset):
    """
        Implement a dataset class providing the image and it's corresponding text
    """
    def __init__(self, images, labels, imagenames, shuffle=None, normalization=None):
        """

        :param images: image datasets [nums, H, W, C] 4D ndarray
        :param labels: label dataset [nums, :] 2D ndarray
        :param shuffle: if need shuffle the dataset, 'once_prior_train' represent shuffle only once before training
                        'every_epoch' represent shuffle the data every epoch
        :param imagenames:
        :param normalization: if need do normalization to the dataset,
                              'None': no any normalization
                              'divide_255': divide all pixels by 255
                              'divide_256': divide all pixels by 256
        """
        super(TextDataset, self).__init__()

        self.__normalization = normalization
        if self.__normalization not in [None, 'divide_255', 'divide_256']:
            raise ValueError('normalization parameter wrong')
        self.__images = self.normalize_images(images, self.__normalization)

        self.__labels = labels
        self.__imagenames = imagenames
        self._epoch_images = copy.deepcopy(self.__images)
        self._epoch_labels = copy.deepcopy(self.__labels)
        self._epoch_imagenames = copy.deepcopy(self.__imagenames)

        self.__shuffle = shuffle
        if self.__shuffle not in [None, 'once_prior_train', 'every_epoch']:
            raise ValueError('shuffle parameter wrong')
        if self.__shuffle == 'every_epoch' or 'once_prior_train':
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = self.shuffle_images_labels(
                self._epoch_images, self._epoch_labels, self._epoch_imagenames)

        self.__batch_counter = 0
        return

    @property
    def num_examples(self):
        """

        :return:
        """
        assert self.__images.shape[0] == self.__labels.shape[0]
        return self.__labels.shape[0]

    @property
    def images(self):
        """

        :return:
        """
        return self._epoch_images

    @property
    def labels(self):
        """

        :return:
        """
        return self._epoch_labels

    @property
    def imagenames(self):
        """

        :return:
        """
        return self._epoch_imagenames

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        start = self.__batch_counter * batch_size
        end = (self.__batch_counter + 1) * batch_size
        self.__batch_counter += 1
        images_slice = self._epoch_images[start:end]
        labels_slice = self._epoch_labels[start:end]
        imagenames_slice = self._epoch_imagenames[start:end]
        # if overflow restart from the begining
        if images_slice.shape[0] != batch_size:
            self.__start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice, imagenames_slice

    def __start_new_epoch(self):
        """

        :return:
        """
        self.__batch_counter = 0

        if self.__shuffle == 'every_epoch':
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = self.shuffle_images_labels(
                self._epoch_images, self._epoch_labels, self._epoch_imagenames)
        else:
            pass
        return


class TextDataProvider(object):
    """
        Implement the text data provider for training and testing the shadow net
    """
    def __init__(self, dataset_dir, annotation_name, validation_set=None, validation_split=None, shuffle=None,
                 normalization=None, input_size: Tuple[int, int]=None):
        """

        :param dataset_dir: Directory with all data.
        :param annotation_name: Annotations file name
        :param validation_set: See `validation_split`
        :param validation_split: `float` or None. If a float, ratio of training data which will will be used as
                                 validation data. If None and if 'validation set' == True, the validation set will be a
                                  copy of the test set.
        :param shuffle: Set to 'once_prior_train' to shuffle the data once before training, 'every_epoch' to shuffle
                        every epoch. None to disable shuffling
        :param normalization: if need do normalization to the dataset,
                              'None': no any normalization
                              'divide_255': divide all pixels by 255
                              'divide_256': divide all pixels by 256
                              'by_chanels': subtract the mean and divide by the standard deviation in each channel
        :param input_size: Target size to which all images will be resized.
        """
        self.__input_size = input_size if input_size is not None else config.cfg.ARCH.INPUT_SIZE
        self.__dataset_dir = dataset_dir
        self.__validation_split = validation_split
        self.__shuffle = shuffle
        self.__normalization = normalization
        self.__train_dataset_dir = ops.join(self.__dataset_dir, 'Train')
        self.__test_dataset_dir = ops.join(self.__dataset_dir, 'Test')

        assert ops.exists(dataset_dir)
        assert ops.exists(self.__train_dataset_dir)
        assert ops.exists(self.__test_dataset_dir)

        def make_datasets(dir: str, split: float=None) -> Tuple[TextDataset, Union[TextDataset, None]]:
            """ Helper function to split data and create TextDatasets
            TODO: maybe shuffle before splitting?
             :param dir: Directory with all data
             :param split: take this fraction of the data for the second TextDataset
            """
            annotation_path = ops.join(dir, annotation_name)
            assert ops.exists(annotation_path)

            with open(annotation_path, 'r', encoding='utf-8') as fd:
                info = np.array(list(filter(lambda x: len(x) == 2,  # discard bogus entries with no label
                                            [line.strip().split(maxsplit=1) for line in fd.readlines()])))

                images_orig = [cv2.imread(ops.join(dir, imgname), cv2.IMREAD_COLOR) for imgname in info[:, 0]]
                assert not any(map(lambda x: x is None, images_orig)),\
                    "Could not read some images. Check for whitespace in file names or invalid files"
                images = np.array([cv2.resize(img, tuple(self.__input_size)) for img in images_orig])
                labels = info[:, 1]
                imagenames = np.array([ops.basename(imgname) for imgname in info[:, 0]])

            if split is None:
                return TextDataset(images, labels, imagenames, shuffle=shuffle, normalization=normalization), None
            else:
                split_idx = int(images.shape[0] * (1.0 - split))
                return TextDataset(images[:split_idx], labels[:split_idx], imagenames[:split_idx],
                                   shuffle=shuffle, normalization=normalization), \
                       TextDataset(images[split_idx:], labels[split_idx:], imagenames[split_idx:],
                                   shuffle=shuffle, normalization=normalization)

        self.test, _ = make_datasets(self.__test_dataset_dir)

        if validation_set is None:
            self.train, _ = make_datasets(self.__train_dataset_dir)
        else:
            if validation_split is None:
                self.validation = self.test
            elif isinstance(validation_split, float) and (0.0 <= validation_split <= 1.0):
                if validation_split > 0.5:
                    print("validation_split suspiciously high: %.2f" % validation_split)
                self.train, self.validation = make_datasets(self.__train_dataset_dir, validation_split)
            else:
                raise ValueError("Expected validation_split to be a float between 0 and 1.")

    def __str__(self):
        provider_info = 'Dataset_dir {:s} contains {:d} training, {:d} validation and {:d} testing images'.\
            format(self.__dataset_dir, self.train.num_examples, self.validation.num_examples, self.test.num_examples)
        return provider_info

    @property
    def input_size(self):
        """ Size to which images are rescaled before training and testing.

        :return:
        """
        return self.__input_size

    @property
    def dataset_dir(self):
        """

        :return:
        """
        return self.__dataset_dir

    @property
    def train_dataset_dir(self):
        """

        :return:
        """
        return self.__train_dataset_dir

    @property
    def test_dataset_dir(self):
        """

        :return:
        """
        return self.__test_dataset_dir
