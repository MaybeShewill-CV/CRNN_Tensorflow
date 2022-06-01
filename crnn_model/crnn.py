#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-21 下午6:39
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : crnn_net.py
# @IDE: PyCharm Community Edition
"""
Implement the crnn model mentioned in An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition paper
"""
import time

import numpy as np
import tensorflow as tf
from tensorflow_core.contrib import rnn

from crnn_model import cnn_basenet
from crnn_model import loss
from local_utils import config_utils


class CrnnNet(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        :return:
        """
        super(CrnnNet, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._cfg = cfg

        self._hidden_nums = self._cfg.MODEL.CRNN.HIDDEN_UNITS
        self._layers_nums = self._cfg.MODEL.CRNN.HIDDEN_LAYERS
        self._num_classes = self._cfg.DATASET.NUM_CLASSES
        self._seq_length = self._cfg.MODEL.CRNN.SEQ_LENGTH

        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._enable_dropout = self._cfg.TRAIN.DROPOUT.ENABLE
        if self._enable_dropout:
            self._dropout_keep_prob = self._cfg.TRAIN.DROPOUT.KEEP_PROB

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _conv_stage(self, inputdata, out_dims, name):
        """
        Standard VGG convolutional stage: 2d conv, relu, and maxpool
        :param inputdata: 4D tensor batch x width x height x channels
        :param out_dims: number of output channels / filters
        :return: the maxpooled output of the stage
        """
        with tf.variable_scope(name_or_scope=name):
            conv = self.conv2d(
                inputdata=inputdata, out_channel=out_dims,
                kernel_size=3, stride=1, use_bias=True, name='conv'
            )
            bn = self.layerbn(
                inputdata=conv, is_training=self._is_training, name='bn'
            )
            relu = self.relu(
                inputdata=bn, name='relu'
            )
            max_pool = self.maxpooling(
                inputdata=relu, kernel_size=2, stride=2, name='max_pool'
            )
        return max_pool

    def _feature_sequence_extraction(self, inputdata, name):
        """
        Implements section 2.1 of the paper: "Feature Sequence Extraction"
        :param inputdata: eg. batch*32*100*3 NHWC format
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            conv1 = self._conv_stage(
                inputdata=inputdata, out_dims=64, name='conv1'
            )
            conv2 = self._conv_stage(
                inputdata=conv1, out_dims=128, name='conv2'
            )
            conv3 = self.conv2d(
                inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3'
            )
            bn3 = self.layerbn(
                inputdata=conv3, is_training=self._is_training, name='bn3'
            )
            relu3 = self.relu(
                inputdata=bn3, name='relu3'
            )
            conv4 = self.conv2d(
                inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4'
            )
            bn4 = self.layerbn(
                inputdata=conv4, is_training=self._is_training, name='bn4'
            )
            relu4 = self.relu(
                inputdata=bn4, name='relu4')
            max_pool4 = self.maxpooling(
                inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID', name='max_pool4'
            )
            conv5 = self.conv2d(
                inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5'
            )
            bn5 = self.layerbn(
                inputdata=conv5, is_training=self._is_training, name='bn5'
            )
            relu5 = self.relu(
                inputdata=bn5, name='bn5'
            )
            conv6 = self.conv2d(
                inputdata=relu5, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv6'
            )
            bn6 = self.layerbn(
                inputdata=conv6, is_training=self._is_training, name='bn6'
            )
            relu6 = self.relu(
                inputdata=bn6, name='relu6'
            )
            max_pool6 = self.maxpooling(
                inputdata=relu6, kernel_size=[2, 1], stride=[2, 1], name='max_pool6'
            )
            conv7 = self.conv2d(
                inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7'
            )
            bn7 = self.layerbn(
                inputdata=conv7, is_training=self._is_training, name='bn7'
            )
            relu7 = self.relu(
                inputdata=bn7, name='bn7'
            )

        return relu7

    def _map_to_sequence(self, inputdata, name):
        """
        Implements the map to sequence part of the network.
        This is used to convert the CNN feature map to the sequence used in the stacked LSTM layers later on.
        Note that this determines the length of the sequences that the LSTM expects
        :param inputdata:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shape = inputdata.get_shape().as_list()
            assert shape[1] == 1  # H of the feature map must equal to 1
            ret = self.squeeze(inputdata=inputdata, axis=1, name='squeeze')
        return ret

    def _sequence_label(self, inputdata, name):
        """
        Implements the sequence label part of the network
        :param inputdata:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]
            # Backward direction cells
            bw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, inputdata,
                dtype=tf.float32
            )
            # stack_lstm_layer, _, _ = tf.nn.bidirectional_dynamic_rnn(
            #     fw_cell_list, bw_cell_list, inputdata,
            #     dtype=tf.float32
            # )
            if self._enable_dropout:
                stack_lstm_layer = tf.cond(
                    self._is_training,
                    true_fn=lambda: self.dropout(
                        inputdata=stack_lstm_layer,
                        keep_prob=self._dropout_keep_prob,
                        name='sequence_dropout_train'
                    ),
                    false_fn=lambda: tf.identity(stack_lstm_layer, name='sequence_dropout_test')
                )
            shape = tf.shape(stack_lstm_layer)
            rnn_reshaped = tf.reshape(stack_lstm_layer, [shape[0] * shape[1], shape[2]])

            w = tf.get_variable(
                name='w',
                shape=[512, self._num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                trainable=True
            )

            # Doing the affine projection
            logits = tf.matmul(rnn_reshaped, w, name='logits')

            logits = tf.reshape(logits, [shape[0], shape[1], self._num_classes], name='logits_reshape')

            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, [1, 0, 2], name='transpose_time_major')  # [width, batch, n_classes]

        return rnn_out, raw_pred

    def _build_net(self, input_tensor, name, reuse):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first apply the cnn feature extraction stage
            cnn_out = self._feature_sequence_extraction(
                inputdata=input_tensor, name='feature_extraction_module'
            )
            # second apply the map to sequence stage
            sequence = self._map_to_sequence(
                inputdata=cnn_out, name='map_to_sequence_module'
            )
            # third apply the sequence label stage
            net_out, raw_pred = self._sequence_label(
                inputdata=sequence, name='sequence_rnn_module'
            )

        return net_out

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            logits = self._build_net(
                input_tensor=input_tensor,
                name='inference',
                reuse=reuse
            )

        return logits

    def compute_loss(self, input_tensor, label, name, reuse=False):
        """

        :param input_tensor:
        :param label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            logits = self._build_net(
                input_tensor=input_tensor,
                name='inference',
                reuse=reuse
            )

            with tf.variable_scope('crnn_loss', reuse=reuse):
                ret = self._loss_func(
                    logits=logits,
                    labels=label,
                    seq_length=self._seq_length,
                    batch_size=input_tensor.get_shape().as_list()[0]
                )
        return ret


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return CrnnNet(phase=phase, cfg=cfg)


def _stats_graph(graph):
    """

    :param graph:
    :return:
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {}; Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    return


def _inference_time_profile():
    """

    :return:
    """
    tf.reset_default_graph()
    cfg = config_utils.get_config(config_file_path='./config/synth_chinese_dataset.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 32, 280, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[10], name='test_label')
    test_label_tensor = tf.contrib.layers.dense_to_sparse(test_label_tensor)
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='CRNN',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='CRNN', reuse=True)
    print(test_result)
    print(tmp_logits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_input = np.random.random((1, 32, 280, 3)).astype(np.float32)
        t_start = time.time()
        loop_times = 100
        for i in range(loop_times):
            _ = sess.run(tmp_logits, feed_dict={test_input_tensor: test_input})
        t_cost = time.time() - t_start
        print('Cost time: {:.5f}s'.format(t_cost / loop_times))
        print('Inference time: {:.5f} fps'.format(loop_times / t_cost))

    print('Complete')


def _model_profile():
    """

    :return:
    """
    tf.reset_default_graph()
    cfg = config_utils.get_config(config_file_path='./config/synth_chinese_dataset.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 32, 280, 3], name='test_input')
    model = get_model(phase='test', cfg=cfg)
    _ = model.inference(input_tensor=test_input_tensor, name='CRNN', reuse=False)

    with tf.Session() as sess:
        _stats_graph(sess.graph)

    print('Complete')


if __name__ == '__main__':
    """
    main func
    """
    _model_profile()

    _inference_time_profile()
