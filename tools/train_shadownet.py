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
import math
import argparse

import tensorflow as tf
import glog as logger
import numpy as np

from crnn_model import crnn_net
from local_utils import evaluation_tools
from config import global_config
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_fast_tools

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
    parser.add_argument('-e', '--decode_outputs', type=args_str2bool, default=False,
                        help='Activate decoding of predictions during training (slow!)')
    parser.add_argument('-m', '--multi_gpus', type=args_str2bool, default=False,
                        nargs='?', const=True, help='Use multi gpus to train')

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


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def compute_net_gradients(images, labels, net, optimizer=None, is_net_first_initialized=False):
    """
    Calculate gradients for single GPU
    :param images: images for training
    :param labels: labels corresponding to images
    :param net: classification model
    :param optimizer: network optimizer
    :param is_net_first_initialized: if the network is initialized
    :return:
    """
    _, net_loss = net.compute_loss(
        inputdata=images,
        labels=labels,
        name='shadow_net',
        reuse=is_net_first_initialized
    )

    if optimizer is not None:
        grads = optimizer.compute_gradients(net_loss)
    else:
        grads = None

    return net_loss, grads


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
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    val_images, val_labels, val_images_paths = val_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )

    # declare crnn net
    shadownet = crnn_net.ShadowNet(
        phase='train',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )
    shadownet_val = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    # set up decoder
    decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

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
        epoch = 0
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            epoch = sess.run(tf.train.get_global_step())

        patience_counter = 1
        cost_history = [np.inf]
        while epoch < train_epochs:
            epoch += 1
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
                    train_predictions, train_labels_sparse, merge_summary_value = sess.run(
                     [optimizer, train_ctc_loss, train_sequence_dist,
                      train_decoded, train_labels, merge_summary_op])

                train_labels_str = decoder.sparse_tensor_to_str(train_labels_sparse)
                train_predictions = decoder.sparse_tensor_to_str(train_predictions[0])
                avg_train_accuracy = evaluation_tools.compute_accuracy(train_labels_str, train_predictions)

                if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, train_ctc_loss_value, train_seq_dist_value, avg_train_accuracy))

                # validation part
                val_ctc_loss_value, val_seq_dist_value, \
                    val_predictions, val_labels_sparse = sess.run(
                     [val_ctc_loss, val_sequence_dist, val_decoded, val_labels])

                val_labels_str = decoder.sparse_tensor_to_str(val_labels_sparse)
                val_predictions = decoder.sparse_tensor_to_str(val_predictions[0])
                avg_val_accuracy = evaluation_tools.compute_accuracy(val_labels_str, val_predictions)

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


def train_shadownet_multi_gpu(dataset_dir, weights_path, char_dict_path, ord_map_dict_path):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :return:
    """
    # prepare dataset information
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
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    val_images, val_labels, val_images_paths = val_dataset.inputs(
        batch_size=CFG.TRAIN.BATCH_SIZE
    )

    # set crnn net
    shadownet = crnn_net.ShadowNet(
        phase='train',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )
    shadownet_val = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    # set average container
    tower_grads = []
    train_tower_loss = []
    val_tower_loss = []
    batchnorm_updates = None
    train_summary_op_updates = None

    # set lr
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=CFG.TRAIN.LEARNING_RATE,
        global_step=global_step,
        decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
        decay_rate=CFG.TRAIN.LR_DECAY_RATE,
        staircase=CFG.TRAIN.LR_STAIRCASE)

    # set up optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # set distributed train op
    with tf.variable_scope(tf.get_variable_scope()):
        is_network_initialized = False
        for i in range(CFG.TRAIN.GPU_NUM):
            with tf.device('/gpu:{:d}'.format(i)):
                with tf.name_scope('tower_{:d}'.format(i)) as _:
                    train_loss, grads = compute_net_gradients(
                        train_images, train_labels, shadownet, optimizer,
                        is_net_first_initialized=is_network_initialized)

                    is_network_initialized = True

                    # Only use the mean and var in the first gpu tower to update the parameter
                    if i == 0:
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)

                    tower_grads.append(grads)
                    train_tower_loss.append(train_loss)
                with tf.name_scope('validation_{:d}'.format(i)) as _:
                    val_loss, _ = compute_net_gradients(
                        val_images, val_labels, shadownet_val, optimizer,
                        is_net_first_initialized=is_network_initialized)
                    val_tower_loss.append(val_loss)

    grads = average_gradients(tower_grads)
    avg_train_loss = tf.reduce_mean(train_tower_loss)
    avg_val_loss = tf.reduce_mean(val_tower_loss)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.TRAIN.MOVING_AVERAGE_DECAY, num_updates=global_step)
    variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all the op needed for training
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # set tensorflow summary
    tboard_save_path = 'tboard/crnn_syn90k_multi_gpu'
    os.makedirs(tboard_save_path, exist_ok=True)

    summary_writer = tf.summary.FileWriter(tboard_save_path)

    avg_train_loss_scalar = tf.summary.scalar(name='average_train_loss',
                                              tensor=avg_train_loss)
    avg_val_loss_scalar = tf.summary.scalar(name='average_val_loss',
                                            tensor=avg_val_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate_scalar',
                                             tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge(
        [avg_train_loss_scalar, learning_rate_scalar] + train_summary_op_updates
    )
    val_merge_summary_op = tf.summary.merge([avg_val_loss_scalar])

    # set tensorflow saver
    saver = tf.train.Saver()
    model_save_dir = 'model/crnn_syn90k_multi_gpu'
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # set sess config
    sess_config = tf.ConfigProto(device_count={'GPU': CFG.TRAIN.GPU_NUM}, allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    logger.info('Global configuration is as follows:')
    logger.info(CFG)

    sess = tf.Session(config=sess_config)

    summary_writer.add_graph(sess.graph)

    with sess.as_default():
        epoch = 0
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/shadownet_model.pb'.format(model_save_dir))

        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)
            epoch = sess.run(tf.train.get_global_step())

        train_cost_time_mean = []
        val_cost_time_mean = []

        while epoch < train_epochs:
            epoch += 1
            # training part
            t_start = time.time()

            _, train_loss_value, train_summary, lr = \
                sess.run(fetches=[train_op,
                                  avg_train_loss,
                                  train_merge_summary_op,
                                  learning_rate])

            if math.isnan(train_loss_value):
                raise ValueError('Train loss is nan')

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)

            summary_writer.add_summary(summary=train_summary,
                                       global_step=epoch)

            # validation part
            t_start_val = time.time()

            val_loss_value, val_summary = \
                sess.run(fetches=[avg_val_loss,
                                  val_merge_summary_op])

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch_Train: {:d} total_loss= {:6f} '
                            'lr= {:6f} mean_cost_time= {:5f}s '.
                            format(epoch + 1,
                                   train_loss_value,
                                   lr,
                                   np.mean(train_cost_time_mean)
                                   ))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                logger.info('Epoch_Val: {:d} total_loss= {:6f} '
                            ' mean_cost_time= {:5f}s '.
                            format(epoch + 1,
                                   val_loss_value,
                                   np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 5000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':

    # init args
    args = init_args()

    if args.multi_gpus:
        logger.info('Use multi gpus to train the model')
        train_shadownet_multi_gpu(
            dataset_dir=args.dataset_dir,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path
        )
    else:
        logger.info('Use single gpu to train the model')
        train_shadownet(
            dataset_dir=args.dataset_dir,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path,
            need_decode=args.decode_outputs
        )
