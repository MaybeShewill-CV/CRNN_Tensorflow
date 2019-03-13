#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os
import os.path as ops
import time
import argparse

import tensorflow as tf
import glog as logger
import numpy as np

from crnn_model import crnn_model
from local_utils import evaluation_tools
from config import global_config
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_tools

CFG = global_config.cfg


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str,
                        help='Directory containing train_features.tfrecords')
    parser.add_argument('-w', '--weights_path', type=str,
                        help='Path to pre-trained weights to continue training')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-e', '--decode_outputs', action='store_true', default=False,
                        help='Activate decoding of predictions during training (slow!)')

    return parser.parse_args()


def train_shadownet(dataset_dir, weights_path, char_dict_path, ord_map_dict_path, need_decode=False):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param need_decode:
    :return:
    """
    # prepare dataset
    train_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        flags='train'
    )
    val_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        flags='val'
    )
    train_images, train_labels, train_images_paths = train_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE,
        num_epochs=1
    )
    val_images, val_labels, val_images_paths = val_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE,
        num_epochs=1
    )

    # declare crnn net
    shadownet = crnn_model.ShadowNet(
        phase='train',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )
    shadownet_val = crnn_model.ShadowNet(
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

    # set up training graph
    with tf.device('/gpu:1'):

        # compute loss and seq distance
        train_inference_ret, train_ctc_loss = shadownet.compute_loss(
            inputdata=train_images,
            labels=train_labels,
            name='shadow_net',
            reuse=False
        )
        val_inference_ret, val_ctc_loss = shadownet_val.compute_loss(
            inputdata=val_images,
            labels=val_labels,
            name='shadow_net',
            reuse=True
        )

        train_decoded, train_log_prob = tf.nn.ctc_beam_search_decoder(
            train_inference_ret,
            CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
            merge_repeated=False
        )
        val_decoded, val_log_prob = tf.nn.ctc_beam_search_decoder(
            val_inference_ret,
            CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
            merge_repeated=False
        )

        train_sequence_dist = tf.reduce_mean(
            tf.edit_distance(tf.cast(train_decoded[0], tf.int32), train_labels),
            name='train_edit_distance'
        )
        val_sequence_dist = tf.reduce_mean(
            tf.edit_distance(tf.cast(val_decoded[0], tf.int32), val_labels),
            name='val_edit_distance'
        )

        # set learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate=CFG.TRAIN.LEARNING_RATE,
            global_step=global_step,
            decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
            decay_rate=CFG.TRAIN.LR_DECAY_RATE,
            staircase=CFG.TRAIN.LR_STAIRCASE)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(
                loss=train_ctc_loss, global_step=global_step)

    # Set tf summary
    tboard_save_dir = 'tboard/crnn_syn90k'
    os.makedirs(tboard_save_dir, exist_ok=True)
    tf.summary.scalar(name='train_ctc_loss', tensor=train_ctc_loss)
    tf.summary.scalar(name='val_ctc_loss', tensor=val_ctc_loss)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    if need_decode:
        tf.summary.scalar(name='train_seq_distance', tensor=train_sequence_dist)
        tf.summary.scalar(name='val_seq_distance', tensor=val_sequence_dist)

    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/crnn_syn90k'
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_dir)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    with sess.as_default():
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        patience_counter = 1
        cost_history = [np.inf]
        for epoch in range(train_epochs):

            # setup early stopping
            if epoch > 1 and CFG.TRAIN.EARLY_STOPPING:
                # We always compare to the first point where cost didn't improve
                if cost_history[-1 - patience_counter] - cost_history[-1] > CFG.TRAIN.PATIENCE_DELTA:
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > CFG.TRAIN.PATIENCE_EPOCHS:
                    logger.info("Cost didn't improve beyond {:f} for {:d} epochs, stopping early.".
                                format(CFG.TRAIN.PATIENCE_DELTA, patience_counter))
                    break

            if need_decode and epoch % 500 == 0:
                # train part
                _, train_ctc_loss_value, train_seq_dist_value, \
                train_predictions, train_labels, merge_summary_value = sess.run(
                    [optimizer, train_ctc_loss, train_sequence_dist,
                     train_decoded, train_labels, merge_summary_op])

                train_labels = decoder.sparse_tensor_to_str(train_labels)
                train_predictions = decoder.sparse_tensor_to_str(train_predictions[0])
                avg_train_accuracy = evaluation_tools.compute_accuracy(train_labels, train_predictions)

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, train_ctc_loss_value, train_seq_dist_value, avg_train_accuracy))

                # validation part
                val_ctc_loss_value, val_seq_dist_value, \
                val_predictions, val_labels = sess.run(
                    [val_ctc_loss, val_sequence_dist, val_decoded, val_labels])

                val_labels = decoder.sparse_tensor_to_str(val_labels)
                val_predictions = decoder.sparse_tensor_to_str(val_predictions[0])
                avg_val_accuracy = evaluation_tools.compute_accuracy(val_labels, val_predictions)

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Val: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, val_ctc_loss_value, val_seq_dist_value, avg_val_accuracy))
            else:
                _, train_ctc_loss_value, merge_summary_value = sess.run(
                    [optimizer, train_ctc_loss, merge_summary_op])

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} cost= {:9f}'.format(epoch + 1, train_ctc_loss_value))

            # record history train ctc loss
            cost_history.append(train_ctc_loss_value)
            # add training sumary
            summary_writer.add_summary(summary=merge_summary_value, global_step=epoch)

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    return np.array(cost_history[1:])  # Don't return the first np.inf


if __name__ == '__main__':

    # init args
    args = init_args()

    train_shadownet(
        dataset_dir=args.dataset_dir,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        need_decode=False
    )
