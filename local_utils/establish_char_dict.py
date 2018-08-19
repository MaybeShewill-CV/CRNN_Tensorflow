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
from typing import Union, List, Set


class CharDictBuilder(object):
    """
        Build and read char dict
    """
    def __init__(self):
        pass

    @staticmethod
    def write_char_dict(origin_char_list: Union[str, List, Set], save_path: str):
        """ Writes the ordinal to char map used in int_to_char to decode predictions and labels.
        The file is read with CharDictBuilder.read_char_dict()

        :param origin_char_list: Either a path to file with character list, one a character per line, or a list or set
                                 of characters
        :param save_path: Destination file, full path.
        """

        if not save_path.endswith('.json'):
            raise ValueError('save path {:s} should be a json file'.format(save_path))

        os.makedirs(ops.split(save_path)[0], exist_ok=True)
        from typing import Iterable
        if isinstance(origin_char_list, str):
            assert ops.exists(origin_char_list), \
                "Character list %s is not a file or could not be found" % origin_char_list
            with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
                chars = (l[0] for l in origin_f.readlines())
        elif isinstance(origin_char_list, Iterable):
            ok = all(map(lambda s: isinstance(s, str) and len(s) == 1, origin_char_list))
            assert ok, "Character list is not a list of strings of length 1"
            chars = origin_char_list
        else:
            raise TypeError("Character list needs to be a file or a list of strings")

        char_dict = {str(ord(c)): c for c in chars}

        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(char_dict, json_f)

    @staticmethod
    def read_char_dict(dict_path: str) -> dict:
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
        """
        assert ops.exists(origin_char_list)

        if not save_path.endswith('.json'):
            raise ValueError('save path {:s} should be a json file'.format(save_path))

        os.makedirs(ops.split(save_path)[0], exist_ok=True)

        char_dict = dict()

        with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
            for index, info in enumerate(origin_f.readlines()):
                char_value = str(ord(info[0]))
                char_key = index
                char_dict[char_key] = char_value

        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(char_dict, json_f)

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path: str) -> dict:
        """

        :param ord_map_dict_path:
        :return:
        """
        assert ops.exists(ord_map_dict_path)

        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)

        return res
