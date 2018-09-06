# Building docker images
From the root directory of the repository, run

```
docker build -t crnn-tf:gpu -f docker/gpu/Dockerfile .
```

```
docker build -t crnn-tf:cpu -f docker/cpu/Dockerfile .
```

