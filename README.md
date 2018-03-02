# CRNN_Tensorflow
Use tensorflow to implement a Deep Neural Network for scene text recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition".You can refer to their paper for details http://arxiv.org/abs/1507.05717. Thanks for the author [Baoguang Shi](https://github.com/bgshih).  
This model consists of a CNN stage, RNN stage and CTC loss for scene text recognition task.

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-8.0, cudnn-6.0 with a GTX-1070 GPU. To install this software you need tensorflow 1.3.0 and other version of tensorflow has not been tested but I think it will be able to work properly in tensorflow above version 1.0. Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on a subset of the [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). During data preparation process the dataset is converted into a tensorflow records which you can find in the data folder.
You can test the trained model on the converted dataset by

```
python tools/test_shadownet.py --dataset_dir data/ --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```
`Expected output is`  
![Test output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_output.png)
If you want to test a single image you can do it by
```
python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```
`Example image_01 is`  
![Example image1](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/text_example_image1.png)  
`Expected output_01 is`  
![Example image1 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image1_output.png)  
`Example image_02 is`  
![Example image2](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2.png)  
`Expected output_02 is`  
![Example image2 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2_output.png) 
`Example image_03 is`  
![Example image3](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese.png)  
`Expected output_03 is`  
![Example image3 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_output.png)
`Example image_04 is`  
![Example image4](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/dmeo_chinese_2.png)  
`Expected output_04 is`  
![Example image4 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_2_ouput.png)

## Train your own model
#### Data Preparation
Firstly you need to store all your image data in a root folder then you need to supply a txt file named sample.txt to specify the relative path to the image data dir and it's corresponding text label. For example

```
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```

Secondly you are supposed to convert your dataset into tensorflow records which can be done by
```
python tools/write_text_features --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir
```
All your training image will be scaled into (32, 100, 3) the dataset will be divided into train, test, validation set and you can change the parameter to control the ratio of them.

#### Train model
The whole training epoches are 40000 in my experiment. I trained the model with a batch size 32, initialized learning rate is 0.1 and decrease by multiply 0.1 every 10000 epochs. For more training parameters information you can check the global_configuration/config.py for details. To train your own model by

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords
```
You can also continue the training process from the snapshot by
```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```
After several times of iteration you can check the log file in logs folder you are supposed to see the following contenent
![Training log](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_log.png)
The seq distance is computed by calculating the distance between two saparse tensor so the lower the accuracy value is the better the model performs.The train accuracy is computed by calculating the character-wise precision between the prediction and the ground truth so the higher the better the model performs.

During my experiment the `loss` drops as follows  
![Training loss](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_loss.png)
The `distance` between the ground truth and the prediction drops as follows  
![Sequence distance](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/seq_distance.png)

## Experiment
The accuracy during training process rises as follows  
![Training accuracy](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/training_accuracy.md)

## TODO
The model is trained on a subet of [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). So i will train a new model on the whold dataset to get a more robust model.The crnn model needs large of training data to get a rubust model.
