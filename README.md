# CRNN_Tensorflow
This is a TensorFlow implementation of a Deep Neural Network for scene text recognition. It is  mainly based on the paper 
["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"](http://arxiv.org/abs/1507.05717). 
You can refer to the paper for architecture details. Thanks to the author [Baoguang Shi](https://github.com/bgshih).
  
The model consists of a CNN stage extracting features which are fed to an RNN stage (Bi-LSTM) and a CTC loss.

## Installation

This software has been developed on Ubuntu 16.04(x64) using python 3.5 and TensorFlow 1.10. Since it uses some recent
features of TensorFlow it is incompatible with older versions.

The following methods are provided to install dependencies:

### Docker

There are Dockerfiles inside the folder `docker`. Follow the instructions inside `docker/README.md` to build the images.

### Conda

You can create a conda environment with the required dependencies using: 

```
conda env create -f crnntf-env.yml
```

### Pip

Required packages may be installed with

```
pip3 install -r requirements.txt
```

## Testing the pre-trained model
In this repo you will find a model pre-trained on a subset of the [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/)
dataset. You can find the data as TensorFlow records in the `data` folder. The trained model can be tested with

```
python tools/test_shadownet.py --dataset_dir data/ --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```

The expected output is
![Test output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_output.png)

If you want to test a single image you can do it with
```
python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```

### Example images
 
![Example image1](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/text_example_image1.png)  

![Example image1 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image1_output.png)  

![Example image2](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2.png)  

![Example image2 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_example_image2_output.png) 

![Example image3](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese.png)  

![Example image3 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_output.png)

![Example image4](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/dmeo_chinese_2.png)  

![Example image4 output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/chinese_version_debug/data/images/demo_chinese_2_ouput.png)

## Training your own model
#### Data preparation
First you need to store all your image data in a root folder then you need to supply a txt file named `sample.txt` to
specify the image paths (relative to the image data dir) and their corresponding text labels. For example

```
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```

Second you need to convert your dataset into TensorFlow records, as well as extract the character set, with

```
python tools/write_text_features --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir --charset_dir path/to/charset_dir
```

All the training images will be scaled to a fixed size (by default (32, 100, 3)) and the dataset will be divided into
train, test and validation set. Check `global_config/config.py` and run `python tools/write_text_features.py` for options.

#### Training

The original experiment run for 40000 epochs, with a batch size of 32, an initial learning rate of 0.1 and exponential
decay of 0.1 every 10000 epochs. For more training parameters you can check `global_configuration/config.py`.
Then train your model with

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords
```

If you wish, you can add more metrics to the training progress messages with `--decode_outputs`, but this will slow
training down. You can also continue the training process from a snapshot with

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```

After several iterations you can check the tensorboard logs in `logs/`. You should see something like:

![Training log](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_log.png)

The sequence distance is computed by calculating the distance between two sparse tensors so the lower the accuracy value
is the better the model performs. The training accuracy is computed by calculating the character-wise precision between
the prediction and the ground truth so the higher the better the model performs.

Finally, note that it is possible to use multiple config files for different experiments, via the option `--config_file`
to all scripts.


## Experiment

During training in the original experiment the `loss` drops as follows  
![Training loss](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_loss.png)

The distance between the ground truth and the prediction drops as follows  
![Sequence distance](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/seq_distance.png)

The accuracy rises as follows  
![Training accuracy](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/training_accuracy.md)

## TODO
The model is trained on a subset of [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). It would make sense to
train on the whole dataset to get a more robust model, since the crnn model needs a large amount of training data in
order to achieve good performance.
