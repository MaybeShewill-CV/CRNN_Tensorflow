#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-30 下午4:01
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : establish_char_dict.py
# @IDE: PyCharm Community Edition
"""
Establish the char dictionary in order to contain chinese character
"""
import json
import os.path as ops
import os


class CharDictBuilder(object):
    """
        Build and read char dict
    """
    def __init__(self):
        pass

    @staticmethod
    def write_char_dict(origin_char_list, save_path: str):
        """

        :param origin_char_list: Origin char you want to contain a character a line
        :param save_path:
        :return:
        """
        assert ops.exists(origin_char_list)

        if not save_path.endswith('.json'):
            raise ValueError('save path {:s} should be a json file'.format(save_path))

        if not ops.exists(ops.split(save_path)[0]):
            os.makedirs(ops.split(save_path)[0])

        char_dict = dict()

        with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
            for info in origin_f.readlines():
                char_value = info[0]
                char_key = str(ord(char_value))
                char_dict[char_key] = char_value

        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(char_dict, json_f)

        return

    @staticmethod
    def read_char_dict(dict_path):
        """

        :param dict_path:
        :return: a dict with ord(char) as key and char as value
        """
        assert ops.exists(dict_path)

        with open(dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)

        return res

    @staticmethod
    def map_ord_to_index(origin_char_list, save_path):
        """
            Map ord of character in origin char list into index start from 0 in order to meet the output of the DNN
        :param origin_char_list:
        :param save_path:
        :return:
        """
        assert ops.exists(origin_char_list)

        if not save_path.endswith('.json'):
            raise ValueError('save path {:s} should be a json file'.format(save_path))

        if not ops.exists(ops.split(save_path)[0]):
            os.makedirs(ops.split(save_path)[0])

        char_dict = dict()

        with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
            for index, info in enumerate(origin_f.readlines()):
                char_value = str(ord(info[0]))
                char_key = index
                char_dict[char_key] = char_value

        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(char_dict, json_f)

        return

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path):
        """

        :param ord_map_dict_path:
        :return:
        """
        assert ops.exists(ord_map_dict_path)

        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)

        return res
