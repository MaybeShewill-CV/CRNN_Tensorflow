#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 下午3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Test shadow net script
"""
import os.path as ops
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np

from local_utils import data_utils
from crnn_model import crnn_model
from global_configuration import config


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the test tfrecords data')
    parser.add_argument('--weights_path', type=str, help='Where you store the shadow net weights')

    return parser.parse_args()


def test_shadownet(dataset_dir, weights_path, is_vis=True):
    """

    :param dataset_dir:
    :param weights_path:
    :param is_vis:
    :return:
    """
    # Initialize the record decoder
    decoder = data_utils.TextFeatureIO().reader
    images_t, labels_t, imagenames_t = decoder.read_features(
        ops.join(dataset_dir, 'train_feature.tfrecords'), num_epochs=None)
    images_sh, labels_sh, imagenames_sh = tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                                                 batch_size=32, capacity=1000+32*2, min_after_dequeue=2,
                                                                 num_threads=4)

    images_sh = tf.cast(x=images_sh, dtype=tf.float32)

    # build shadownet
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=images_sh)

    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(32), merge_repeated=False)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start predicting ......')
        predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
        imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
        imagenames = [tmp.decode('utf-8') for tmp in imagenames]
        preds, preds_res = decoder.sparse_tensor_to_str(predictions[0])
        gt_labels, gt_res = decoder.sparse_tensor_to_str(labels)
        for index, image in enumerate(images):
            # prediction = decoder.sparse_tensor_to_str(predictions[index])
            # label = decoder.sparse_tensor_to_str(labels[index])
            print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                imagenames[index], gt_res[index], preds_res[index]))
            if is_vis:
                plt.imshow(image[:, :, (2, 1, 0)])
                plt.show()

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test shadow net
    test_shadownet(args.dataset_dir, args.weights_path)
