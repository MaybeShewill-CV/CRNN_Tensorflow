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
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse

from crnn_model import crnn_model
from local_utils import data_utils, log_utils
from global_configuration import config


logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='Path to dir containing train/test data and annotation files.')
    parser.add_argument('-w', '--weights_path', type=str, help='Path to pretrained weights.')
    parser.add_argument('-j', '--num_threads', type=int, default=int(os.cpu_count()/2),
                        help='Number of threads to use in batch shuffling')

    return parser.parse_args()


def train_shadownet(dataset_dir, weights_path=None, num_threads=4):
    """

    :param dataset_dir:
    :param weights_path:
    :param num_threads: Number of threads to use in tf.train.shuffle_batch
    :return:
    """
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO().reader
    images, labels, imagenames = decoder.read_features(ops.join(dataset_dir, 'train_feature.tfrecords'),
                                                       num_epochs=None)
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames], batch_size=config.cfg.TRAIN.BATCH_SIZE,
        capacity=1000 + 2*config.cfg.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=num_threads)

    inputdata = tf.cast(x=inputdata, dtype=tf.float32)

    # initialise the net model
    shadownet = crnn_model.ShadowNet(phase='Train',
                                     hidden_nums=config.cfg.ARCH.HIDDEN_UNITS,
                                     layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,
                                     num_classes=len(decoder.char_dict))

    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out,
                                         sequence_length=config.cfg.ARCH.SEQ_LENGTH*np.ones(config.cfg.TRAIN.BATCH_SIZE)))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,
                                                      config.cfg.ARCH.SEQ_LENGTH*np.ones(config.cfg.TRAIN.BATCH_SIZE),
                                                      merge_repeated=False)

    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)

    # Set tf summary
    tboard_save_path = 'tboard/shadownet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/shadownet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = config.cfg.TRAIN.EPOCHS

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
            _, c, seq_distance, preds, gt_labels, summary = sess.run(
                [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op])

            # calculate the precision
            preds = decoder.sparse_tensor_to_str(preds[0])
            gt_labels = decoder.sparse_tensor_to_str(gt_labels)

            accuracy = []

            for index, gt_label in enumerate(gt_labels):
                pred = preds[index]
                total_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            #
            if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                    epoch + 1, c, seq_distance, accuracy))

            summary_writer.add_summary(summary=summary, global_step=epoch)
            saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if not ops.exists(args.dataset_dir):
        raise ValueError('{:s} doesn\'t exist'.format(args.dataset_dir))

    train_shadownet(args.dataset_dir, args.weights_path)
    print('Done')
