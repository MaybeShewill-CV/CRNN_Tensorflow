# Building docker images
From the root directory of the repository, run

```
docker build crnn-tf:gpu -f docker/gpu/Dockerfile .
```

```
docker build crnn-tf:cpu -f docker/cpu/Dockerfile .
```

