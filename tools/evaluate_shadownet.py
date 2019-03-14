#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 下午3:56
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : evaluate_shadownet.py
# @IDE: PyCharm Community Edition
"""
Evaluate the crnn model on the synth90k test dataset
"""
import argparse
import os.path as ops
import math
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glog as log

from crnn_model import crnn_model
from config import global_config
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_tools
from local_utils import evaluation_tools


CFG = global_config.cfg


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-w', '--weights_path', type=str, required=True,
                        help='Path to pre-trained weights')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=False,
                        help='Whether to display images')
    parser.add_argument('-p', '--process_all', type=args_str2bool, nargs='?', const=True,
                        help='Whether to process all test dataset')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def evaluate_shadownet(dataset_dir, weights_path, char_dict_path,
                       ord_map_dict_path, is_visualize=False,
                       is_process_all_data=False):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_visualize:
    :param is_process_all_data:
    :return:
    """
    # prepare dataset
    test_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        flags='test'
    )
    test_images, test_labels, test_images_paths = test_dataset.inputs(
        batch_size=CFG.TEST.BATCH_SIZE,
        num_epochs=1
    )

    # set up test sample count
    if is_process_all_data:
        log.info('Start computing test dataset sample counts')
        t_start = time.time()
        test_sample_count = test_dataset.sample_counts()
        log.info('Computing test dataset sample counts finished, cost time: {:.5f}'.format(time.time() - t_start))
        num_iterations = int(math.ceil(test_sample_count / CFG.TEST.BATCH_SIZE))
    else:
        num_iterations = 1

    # declare crnn net
    shadownet = crnn_model.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )
    # set up decoder
    decoder = tf_io_pipline_tools.TextFeatureIO(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    ).reader

    # compute inference result
    test_inference_ret = shadownet.inference(
        inputdata=test_images,
        name='shadow_net',
        reuse=False
    )
    test_decoded, test_log_prob = tf.nn.ctc_beam_search_decoder(
        test_inference_ret,
        CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TEST.BATCH_SIZE),
        beam_width=1,
        merge_repeated=False
    )

    # recover image from [-1.0, 1.0] ---> [0.0, 255.0]
    test_images = tf.multiply(tf.add(test_images, 1.0), 127.5, name='recoverd_test_images')

    # Set saver configuration
    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        log.info('Start predicting...')

        accuracy = 0

        while True:
            try:

                for epoch in range(num_iterations):
                    test_predictions_value, test_images_value, test_labels_value, \
                    test_images_paths_value = sess.run(
                        [test_decoded, test_images, test_labels, test_images_paths]
                    )
                    test_images_paths_value = np.reshape(
                        test_images_paths_value,
                        newshape=test_images_paths_value.shape[0]
                    )
                    test_images_paths_value = [tmp.decode('utf-8') for tmp in test_images_paths_value]
                    test_images_names_value = [ops.split(tmp)[0] for tmp in test_images_paths_value]
                    test_labels_value = decoder.sparse_tensor_to_str(test_labels_value)
                    test_predictions_value = decoder.sparse_tensor_to_str(test_predictions_value[0])

                    accuracy += evaluation_tools.compute_accuracy(
                        test_labels_value, test_predictions_value, display=False
                    )

                    for index, test_image in enumerate(test_images_value):
                        print('Predict {:s} image with gt label: {:s} **** predicted label: {:s}'.format(
                            ops.split(test_images_names_value[index])[1],
                            test_labels_value[index],
                            test_predictions_value[index]))

                        # avoid accidentally displaying for the whole dataset
                        if is_visualize:
                            plt.imshow(np.array(test_image, np.uint8)[:, :, (2, 1, 0)])
                            plt.show()
            except tf.errors.OutOfRangeError:
                print('End of tfrecords sequence')
                break
            except Exception as err:
                print(err)
                break

        # we compute a mean of means, so we need the sample sizes to be constant
        # (BATCH_SIZE) for this to equal the actual mean
        accuracy /= num_iterations
        log.info('Mean test accuracy is {:5f}'.format(accuracy))


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    evaluate_shadownet(
        dataset_dir=args.dataset_dir,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        is_visualize=args.visualize,
        is_process_all_data=args.process_all
    )
