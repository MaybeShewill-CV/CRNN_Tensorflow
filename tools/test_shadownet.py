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
import importlib
import os
import os.path as ops
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math

from local_utils import data_utils
from crnn_model import crnn_model
from easydict import EasyDict


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Path to test tfrecords data')
    parser.add_argument('-w', '--weights_path', type=str, required=True,
                        help='Path to pre-trained weights')
    parser.add_argument('-c', '--charset_dir', type=str, default='data/char_dict',
                        help='Path to dir where character sets for the dataset were stored')
    parser.add_argument('-n', '--num_classes', type=int, required=True,
                        help='Force number of character classes to this number. '
                             'Use 37 to run with the demo data. '
                             'Set to 0 for auto (read from files in charset_dir)')
    parser.add_argument('-f', '--config_file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-v', '--visualise', type=bool, default=False,
                        help='Whether to display images')
    parser.add_argument('-b', '--one_batch', default=False,
                        action='store_true', help='Test only one batch of the dataset')
    parser.add_argument('-j', '--num_threads', type=int,
                        default=int(os.cpu_count() / 2),
                        help='Number of threads to use in batch shuffling')

    return parser.parse_args()


def test_shadownet(dataset_dir: str, charset_dir: str, weights_path: str, cfg: EasyDict, is_vis: bool,
                   process_all_data: bool=True, num_threads: int=4, num_classes: int=0):
    """

    :param dataset_dir: Path to Train and Test directories
    :param charset_dir: Path to char_dict.json and ord_map.json (generated with write_text_features.py)
    :param weights_path: Path to stored weights
    :param cfg: configuration EasyDict (e.g. global_config.config.cfg)
    :param is_vis: whether to visualise the result
    :param process_all_data:
    :param num_threads: Number of threads for tf.train.(shuffle_)batch
    :param num_classes: Number of different characters in the dataset
    """
    # Initialize the record decoder
    decoder = data_utils.TextFeatureIO(char_dict_path=ops.join(charset_dir, 'char_dict.json'),
                                       ord_map_dict_path=ops.join(charset_dir, 'ord_map.json')).reader
    images_t, labels_t, imagenames_t = decoder.read_features(
        ops.join(dataset_dir, 'test_feature.tfrecords'), num_epochs=None)
    if not process_all_data:
        images_sh, labels_sh, imagenames_sh = tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                                                     batch_size=cfg.TEST.BATCH_SIZE,
                                                                     capacity=1000 + 2*cfg.TEST.BATCH_SIZE,
                                                                     min_after_dequeue=2, num_threads=num_threads)
    else:
        images_sh, labels_sh, imagenames_sh = tf.train.batch(tensors=[images_t, labels_t, imagenames_t],
                                                             batch_size=cfg.TEST.BATCH_SIZE,
                                                             capacity=1000 + 2 * cfg.TEST.BATCH_SIZE,
                                                             num_threads=num_threads)

    images_sh = tf.cast(x=images_sh, dtype=tf.float32)

    # build shadownet
    num_classes = len(decoder.char_dict) + 1 if num_classes == 0 else num_classes
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=cfg.ARCH.HIDDEN_UNITS,
                               layers_nums=cfg.ARCH.HIDDEN_LAYERS,
                               num_classes=num_classes)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=images_sh)

    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out,
                                               cfg.ARCH.SEQ_LENGTH * np.ones(cfg.TEST.BATCH_SIZE),
                                               merge_repeated=False)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(ops.join(dataset_dir, 'test_feature.tfrecords')):
        test_sample_count += 1
    loops_nums = int(math.ceil(test_sample_count / cfg.TEST.BATCH_SIZE))
    # loops_nums = 100

    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start predicting ......')
        if not process_all_data:
            predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
            imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
            imagenames = [tmp.decode('utf-8') for tmp in imagenames]
            preds_res = decoder.sparse_tensor_to_str(predictions[0])
            gt_res = decoder.sparse_tensor_to_str(labels)

            accuracy = []

            for index, gt_label in enumerate(gt_res):
                pred = preds_res[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Mean test accuracy is {:5f}'.format(accuracy))

            for index, image in enumerate(images):
                print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                    imagenames[index], gt_res[index], preds_res[index]))
                if is_vis:
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.show()
        else:
            accuracy = []
            for epoch in range(loops_nums):
                predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
                imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
                imagenames = [tmp.decode('utf-8') for tmp in imagenames]
                preds_res = decoder.sparse_tensor_to_str(predictions[0])
                gt_res = decoder.sparse_tensor_to_str(labels)

                for index, gt_label in enumerate(gt_res):
                    pred = preds_res[index]
                    totol_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / totol_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)

                for index, image in enumerate(images):
                    print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                        imagenames[index], gt_res[index], preds_res[index]))
                    # if is_vis:
                    #     plt.imshow(image[:, :, (2, 1, 0)])
                    #     plt.show()

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Test accuracy is {:5f}'.format(accuracy))

        coord.request_stop()
        coord.join(threads=threads)


if __name__ == '__main__':

    args = init_args()

    config = {}  # Silence PyCharm's checks
    if args.config_file:
        # Remove extension in case the user gave it
        args.config_file = os.path.splitext(args.config_file)[0]
        path = os.path.abspath(os.path.dirname(args.config_file))
        module = os.path.basename(args.config_file)
    else:
        path = "."
        module = "global_configuration.config"

    try:
        save_path = sys.path
        sys.path = [path] + sys.path  # Search here first
        print("Importing configuration {:s} from {:s}".format(module, path))
        config = importlib.import_module(module)
        sys.path = save_path
    except:
        print("Configuration file not found or invalid")
        exit(1)

    test_shadownet(dataset_dir=args.dataset_dir, charset_dir=args.charset_dir,
                   weights_path=args.weights_path, cfg=config.cfg, process_all_data=not args.one_batch,
                   is_vis=args.visualise, num_threads=args.num_threads, num_classes=args.num_classes)
