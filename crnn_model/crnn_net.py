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
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from crnn_model import cnn_basenet
from config import global_config

CFG = global_config.cfg


class ShadowNet(cnn_basenet.CNNBaseModel):
    """
        Implement the crnn model for squence recognition
    """
    def __init__(self, phase, hidden_nums, layers_nums, num_classes):
        """

        :param phase: 'Train' or 'Test'
        :param hidden_nums: Number of hidden units in each LSTM cell (block)
        :param layers_nums: Number of LSTM cells (blocks)
        :param num_classes: Number of classes (different symbols) to detect
        """
        super(ShadowNet, self).__init__()
        if phase == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._hidden_nums = hidden_nums
        self._layers_nums = layers_nums
        self._num_classes = num_classes
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    def _conv_stage(self, inputdata, out_dims, name):
        """ Standard VGG convolutional stage: 2d conv, relu, and maxpool

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
        """ Implements section 2.1 of the paper: "Feature Sequence Extraction"

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
        """ Implements the map to sequence part of the network.

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
        """ Implements the sequence label part of the network

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
            stack_lstm_layer = self.dropout(
                inputdata=stack_lstm_layer,
                keep_prob=0.5,
                is_training=self._is_training,
                name='sequence_drop_out'
            )

            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]

            shape = tf.shape(stack_lstm_layer)
            rnn_reshaped = tf.reshape(stack_lstm_layer, [shape[0] * shape[1], shape[2]])

            w = tf.get_variable(
                name='w',
                shape=[hidden_nums, self._num_classes],
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

    def inference(self, inputdata, name, reuse=False):
        """
        Main routine to construct the network
        :param inputdata:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            # first apply the cnn feature extraction stage
            cnn_out = self._feature_sequence_extraction(
                inputdata=inputdata, name='feature_extraction_module'
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

    def compute_loss(self, inputdata, labels, name, reuse):
        """

        :param inputdata:
        :param labels:
        :return:
        """

        inference_ret = self.inference(
            inputdata=inputdata, name=name, reuse=reuse
        )

        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=labels, inputs=inference_ret,
                sequence_length=CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE)
            ),
            name='ctc_loss'
        )

        return inference_ret, loss
