#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-12 下午9:03
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : evaluation_tools.py
# @IDE: PyCharm
"""
Some evaluation tools
"""
import numpy as np


def compute_accuracy(ground_truth, predictions, display=False):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :return:
    """
    accuracy = []

    for index, label in enumerate(ground_truth):
        prediction = predictions[index]
        total_count = len(label)
        correct_count = 0
        try:
            for i, tmp in enumerate(label):
                if tmp == prediction[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / total_count)
            except ZeroDivisionError:
                if len(prediction) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)

    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    if display:
        print('Mean accuracy is {:5f}'.format(accuracy))

    return accuracy
