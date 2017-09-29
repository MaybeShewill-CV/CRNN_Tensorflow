#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 上午10:46
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : decode_cifar_10.py
# @IDE: PyCharm Community Edition
"""
Decode the cifar 10 dataset
"""
import pickle
import numpy as np
import os
import copy
import os.path as ops
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


CIFAR_PICKLE_PATH = '/home/baidu/cifar-10-batches-py/data_batch_1'


def decode_cifar10(data_path, save_dir):
    """

    :param data_path:
    :param save_dir:
    :return:
    """
    with open(data_path, 'rb') as fo:
        _dict = pickle.load(fo, encoding='bytes')
    images_data = _dict['data'.encode('utf-8')]
    label_indexs = _dict['labels'.encode('utf-8')]
    filenames = _dict['filenames'.encode('utf-8')]

    for i in range(images_data.shape[0]):
        image = np.zeros([32, 32, 3], np.uint8)
        r_channel = images_data[i][0:32*32]
        g_channel = images_data[i][32*32:32*32*2]
        b_channel = images_data[i][32*32*2:]
        r = np.zeros([32, 32], np.uint8)
        g = np.zeros([32, 32], np.uint8)
        b = np.zeros([32, 32], np.uint8)

        for j in range(32):
            r[j, :] = r_channel[j * 32:j * 32 + 32]
            g[j, :] = g_channel[j * 32:j * 32 + 32]
            b[j, :] = b_channel[j * 32:j * 32 + 32]
        image[:, :, 0] = b
        image[:, :, 1] = g
        image[:, :, 2] = r

        # image = np.reshape(images_data[i, :], newshape=[32, 32, 3]).astype(np.uint8)[:, :, (2, 1, 0)]
        class_name = label_indexs[i]
        filename = filenames[i].decode('utf-8')
        if not ops.exists(ops.join(save_dir, str(class_name))):
            os.makedirs(ops.join(save_dir, str(class_name)))
        image_save_path = ops.join(save_dir, str(class_name))
        image_save_index = len([tmp for tmp in os.listdir(image_save_path) if tmp.endswith('.jpg')]) + 1
        image_save_path = ops.join(image_save_path, '{:s}.jpg'.format(filename[0:-4]))
        cv2.imwrite(image_save_path, image)
        print('Write the {:d} image in label {:d} {:s}'.format(image_save_index, class_name,
                                                               '{:s}.jpg'.format(filename[0:-4])))
    return


def main(data_dir, save_dir):
    """

    :param data_dir:
    :param save_dir:
    :return:
    """
    data_path = [tmp for tmp in os.listdir(data_dir)]

    for data in data_path:
        decode_cifar10(ops.join(data_dir, data), save_dir)
    return


if __name__ == '__main__':
    # main('/home/baidu/DataBase/cifar-10-batches-py', '/home/baidu/DataBase/Image_Retrieval/cifar_10/Test')
    a = ['image/name/text.jpg text', 'a/b/sb.jpg text', [1, 2, 3]]
    b = np.array([[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]])
    c = np.reshape(b, [-1, 3])
    print(b)
    print(c)
