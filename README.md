# Action Recognition with Transformers

**Training a action recognizer with hybrid transformers.**

We will build a Transformer-based model to recognize the actions from videos.we are going to develop hybrid Transformer-based models for action recognition that operate on CNN feature maps.

## Download the dataset
In order to make training time to low, we will be using a subsampled version of the original [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) dataset. download the dataset from [UCF101](https://git.io/JGc31) dataset link.

## Requirements
Before run the code you should run below the lines for installing dependencies
```bash
  pip install tensorflow
  pip install -q git+https://github.com/tensorflow/docs
  pip install imutils
  pip install opencv-python
  pip install matplotlib
  pip install seaborn
```

## Data preparation

We will mostly be following the same data preparation steps in this example, except for
the following changes:

* We took the image size to 128x128 to speed up computation.
* We use [DenseNet121](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
for feature extraction.
* We directly pad shorter videos by zero to length `MAX_SEQ_LENGTH`.
