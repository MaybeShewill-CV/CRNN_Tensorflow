#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-14 下午7:59
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : crnn_model_docker.py
# @IDE: PyCharm
"""
crnn model docker
"""
import numpy as np
import docker
import json
import grpc

IMAGE_NAME = 'image_name'
TAG = 'tag'


class CrnnDocker(object):

    def __init__(self):
        self.client = docker.from_env()

    def get_container(self, client):
        """

        :param client:
        :return:
        """
        container = client.containers.run(
            image=IMAGE_NAME,
            command='python server.py',
            runtime='nvidia',
            environment=["CUDA_VISIBLE_DEVICES=0"],
            ports={'8888/tcp': '8888'},
            detach=True,
            auto_remove=True)

        return container

    def __enter__(self):
        """

        :return:
        """
        self.container = self.get_container(self.client)
        for line in self.container.logs(stream=True):
            if line.strip().find(b'grpc_server_start') >= 0:
                break
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.container.stop()
        print('container has stopped.')

    def run(self, img):
        """

        :param img:
        :return:
        """
        assert isinstance(img, np.ndarray), 'img must be a numpy array.'
        imgstr = img.tobytes()
        shape = json.dumps(img.shape)
        stub = ctpn_pb2_grpc.ModelStub(grpc.insecure_channel('localhost:50051'))
        response = stub.predict(ctpn_pb2.rect_request(img=imgstr, shape=shape))

        return json.loads(response.message)
