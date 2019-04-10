#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-8 下午10:24
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : recongnize_chinese_pdf.py
# @IDE: PyCharm
"""
test the model to recognize the chinese pdf file
"""
import argparse

import cv2
import numpy as np
import tensorflow as tf

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg


def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested',
                        default='data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('--save_path', type=str,
                        help='The output path of recognition result')

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


def split_pdf_image_into_row_image_block(pdf_image):
    """
    split the whole pdf image into row image block
    :param pdf_image: the whole color pdf image
    :return:
    """
    gray_image = cv2.cvtColor(pdf_image, cv2.COLOR_BGR2GRAY)
    binarized_image = cv2.adaptiveThreshold(
        src=gray_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # sum along the row axis
    row_sum = np.sum(binarized_image, axis=1)
    idx_row_sum = np.argwhere(row_sum < row_sum.max())[:, 0]

    split_idx = []

    start_idx = idx_row_sum[0]
    for index, idx in enumerate(idx_row_sum[:-1]):
        if idx_row_sum[index + 1] - idx > 5:
            end_idx = idx
            split_idx.append((start_idx, end_idx))
            start_idx = idx_row_sum[index + 1]
    split_idx.append((start_idx, idx_row_sum[-1]))

    pdf_image_splits = []
    for index in range(len(split_idx)):
        idx = split_idx[index]
        pdf_image_split = pdf_image[idx[0]:idx[1], :, :]
        pdf_image_splits.append(pdf_image_split)

    return pdf_image_splits


def locate_text_area(pdf_image_row_block):
    """
    locate the text area of the image row block
    :param pdf_image_row_block: color pdf image block
    :return:
    """
    gray_image = cv2.cvtColor(pdf_image_row_block, cv2.COLOR_BGR2GRAY)
    binarized_image = cv2.adaptiveThreshold(
        src=gray_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # sum along the col axis
    col_sum = np.sum(binarized_image, axis=0)
    idx_col_sum = np.argwhere(col_sum < col_sum.max())[:, 0]

    start_col = idx_col_sum[0] if idx_col_sum[0] > 0 else 0
    end_col = idx_col_sum[-1]
    end_col = end_col if end_col < pdf_image_row_block.shape[1] else pdf_image_row_block.shape[1] - 1

    return pdf_image_row_block[:, start_col:end_col, :]


def recognize(image_path, weights_path, char_dict_path, ord_map_dict_path, output_path):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param output_path:
    :return:
    """
    # read pdf image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # split pdf image into row block
    pdf_image_row_blocks = split_pdf_image_into_row_image_block(image)

    # locate the text area in each row block
    pdf_image_text_areas = []
    new_heigth = 32
    max_text_area_length = -1
    for index, row_block in enumerate(pdf_image_row_blocks):
        text_area = locate_text_area(row_block)
        text_area_height = text_area.shape[0]
        scale = new_heigth / text_area_height
        max_text_area_length = max(max_text_area_length, int(scale * text_area.shape[1]))
        pdf_image_text_areas.append(text_area)
    new_width = max_text_area_length
    new_width = new_width if new_width > CFG.ARCH.INPUT_SIZE[0] else CFG.ARCH.INPUT_SIZE[0]

    # definite the compute graph
    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, new_heigth, new_width, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

    codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )

    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=int(new_width / 4) * np.ones(1),
        merge_repeated=False,
        beam_width=1
    )

    # config tf saver
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        pdf_recognize_results = []

        for index, pdf_image_text_area in enumerate(pdf_image_text_areas):
            # resize text area into size (None, new_height)
            pdf_image_text_area_height = pdf_image_text_area.shape[0]
            scale = new_heigth / pdf_image_text_area_height
            new_width_tmp = int(scale * pdf_image_text_area.shape[1])
            pdf_image_text_area = cv2.resize(
                pdf_image_text_area, (new_width_tmp, new_heigth),
                interpolation=cv2.INTER_LINEAR)
            # pad text area into size (new_width, new_height) if new_width_tmp < new_width
            if new_width_tmp < new_width:
                pad_area_width = new_width - new_width_tmp
                pad_area = np.zeros(shape=[new_heigth, pad_area_width, 3], dtype=np.uint8) + 255
                pdf_image_text_area = np.concatenate((pdf_image_text_area, pad_area), axis=1)

            pdf_image_text_area = np.array(pdf_image_text_area, np.float32) / 127.5 - 1.0

            preds = sess.run(decodes, feed_dict={inputdata: [pdf_image_text_area]})

            preds = codec.sparse_tensor_to_str(preds[0])

            pdf_recognize_results.append(preds[0])

        output_text = []

        need_tab = True
        for index, pdf_text in enumerate(pdf_recognize_results):
            if need_tab:
                text_console_str = '----     {:s}'.format(pdf_text)
                text_file_str = '     {:s}'.format(pdf_text)
                print(text_console_str)
                output_text.append(text_file_str)
                need_tab = \
                    index < (len(pdf_recognize_results) - 1) and \
                    len(pdf_recognize_results[index + 1]) - len(pdf_text) > 10
            else:
                text_console_str = '---- {:s}'.format(pdf_text)
                text_file_str = ' {:s}'.format(pdf_text)
                print(text_console_str)
                output_text.append(text_file_str)
                need_tab = \
                    index < (len(pdf_recognize_results) - 1) and \
                    len(pdf_recognize_results[index + 1]) - len(pdf_text) > 10

        res = '\n'.join(output_text)

        with open(output_path, 'w') as file:
            file.writelines(res)

    return


if __name__ == '__main__':
    """

    """
    # init images
    args = init_args()

    # detect images
    recognize(
        image_path=args.image_path,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        output_path=args.save_path
    )
