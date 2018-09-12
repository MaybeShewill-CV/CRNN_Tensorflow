#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os
from typing import Tuple

import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse
from easydict import EasyDict

from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from local_utils.log_utils import compute_accuracy
from local_utils.config_utils import load_config

logger = log_utils.init_logger()


def init_args() -> Tuple[argparse.Namespace, EasyDict]:
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-c', '--chardict_dir', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-m', '--model_dir', type=str,
                        help='Directory where to store model checkpoints')
    parser.add_argument('-t', '--tboard_dir', type=str,
                        help='Directory where to store TensorBoard logs')
    parser.add_argument('-f', '--config_file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-e', '--decode_outputs', action='store_true', default=False,
                        help='Activate decoding of predictions during training (slow!)')
    parser.add_argument('-w', '--weights_path', type=str, help='Path to pre-trained weights to continue training')
    parser.add_argument('-j', '--num_threads', type=int, default=int(os.cpu_count()/2),
                        help='Number of threads to use in batch shuffling')

    args = parser.parse_args()

    config = load_config(args.config_file)
    if args.dataset_dir:
        config.cfg.PATH.TFRECORDS_DIR = args.dataset_dir
    if args.chardict_dir:
        config.cfg.PATH.CHAR_DICT_DIR = args.chardict_dir
    if args.model_dir:
        config.cfg.PATH.MODEL_SAVE_DIR = args.model_dir
    if args.tboard_dir:
        config.cfg.PATH.TBOARD_SAVE_DIR = args.tboard_dir

    return args, config.cfg


def train_shadownet(cfg: EasyDict, weights_path: str=None, decode: bool=False, num_threads: int=4):
    """
    :param cfg: configuration EasyDict (e.g. global_config.config.cfg)
    :param weights_path: Path to stored weights
    :param decode: Whether to perform CTC decoding to report progress during training
    :param num_threads: Number of threads to use in tf.train.shuffle_batch
    """
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO(char_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'char_dict.json'),
                                       ord_map_dict_path=ops.join(cfg.PATH.CHAR_DICT_DIR, 'ord_map.json')).reader
    images, labels, imagenames = decoder.read_features(ops.join(cfg.PATH.TFRECORDS_DIR, 'train_feature.tfrecords'),
                                                       num_epochs=None, input_size=cfg.ARCH.INPUT_SIZE,
                                                       input_channels=cfg.ARCH.INPUT_CHANNELS)
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames], batch_size=cfg.TRAIN.BATCH_SIZE,
        capacity=1000 + 2*cfg.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=num_threads)

    inputdata = tf.cast(x=inputdata, dtype=tf.float32)

    # initialise the net model
    shadownet = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=cfg.ARCH.HIDDEN_UNITS,
                                     layers_nums=cfg.ARCH.HIDDEN_LAYERS,
                                     num_classes=len(decoder.char_dict)+1)

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out,
                                         sequence_length=cfg.ARCH.SEQ_LENGTH*np.ones(cfg.TRAIN.BATCH_SIZE)))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,
                                                      cfg.ARCH.SEQ_LENGTH*np.ones(cfg.TRAIN.BATCH_SIZE),
                                                      merge_repeated=False)

    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               cfg.TRAIN.LR_DECAY_STEPS, cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)

    # Set tf summary
    os.makedirs(cfg.PATH.TBOARD_SAVE_DIR, exist_ok=True)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    os.makedirs(cfg.PATH.TBOARD_SAVE_DIR, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(cfg.PATH.MODEL_SAVE_DIR, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(cfg.PATH.TBOARD_SAVE_DIR)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = cfg.TRAIN.EPOCHS

    with sess.as_default():
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(train_epochs):
            if decode:
                _, c, seq_distance, predictions, labels, summary = sess.run(
                    [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op])

                labels = decoder.sparse_tensor_to_str(labels)
                predictions = decoder.sparse_tensor_to_str(predictions[0])
                accuracy = compute_accuracy(labels, predictions)

                if epoch % cfg.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, c, seq_distance, accuracy))

            else:
                _, c, summary = sess.run([optimizer, cost, merge_summary_op])
                if epoch % cfg.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch: {:d} cost= {:9f}'.format(epoch + 1, c))

            summary_writer.add_summary(summary=summary, global_step=epoch)
            saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        coord.request_stop()
        coord.join(threads=threads)


if __name__ == '__main__':
    args, cfg = init_args()

    train_shadownet(cfg=cfg, weights_path=args.weights_path, decode=args.decode_outputs,
                    num_threads=args.num_threads)
    print('Done')
