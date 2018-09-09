# Building docker images
From the root directory of the repository, run

Build the images with GPU version

```
docker build -t crnn-tf:gpu -f docker/gpu/Dockerfile .
```

To run the docker images in container with nvidia docker

```
nvidia-docker run -it --rm -v <path to project>:/app crnn-tf:gpu
```

Build the images with CPU version

```
docker build -t crnn-tf:cpu -f docker/cpu/Dockerfile .
```

To run the docker images in container

```
docker run -it --rm -v <path to project>:/app crnn-tf:cpu
```

