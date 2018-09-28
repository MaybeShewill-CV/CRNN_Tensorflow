FROM tensorflow/tensorflow:1.10.0-gpu-py3

RUN apt update && \
    apt install -y --no-install-recommends \
        python3-tk \
        vim \
        ssh \
        git && \
    rm -rf /var/lib/apt/lists/* 

ADD . /app

WORKDIR /app

RUN pip3 install -r docker/requirements.txt

ENV PYTHONPATH=/app

# TensorBoard
EXPOSE 6006

# JupyterLab
EXPOSE 8888

CMD ["/bin/bash"]

