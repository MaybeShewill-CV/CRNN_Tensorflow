#!/bin/bash

set -eux

docker run \
    --runtime=nvidia \
    --name tfserve_crnn_gpu \
    --publish 8501:8501 \
    --publish 8500:8500 \
    --mount type=bind,source=/tmp/crnn,target=/models/crnn \
    --env MODEL_NAME=crnn \
    --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env TF_CPP_MIN_VLOG_LEVEL=0 \
    --tty \
    tensorflow/serving:latest-gpu
