# BSUV-Net-2.0

python 3.6.9
pytorch 1.3.0
opencv 4.0.1

## Requirements
1. [Python 3.6.9](https://www.python.org/)
2. [PyTorch 1.3](https://pytorch.org/)
3. [OpenCV 4.0.1](https://opencv.org/releases/)

## Dataset
This repository includes a sampled version of the CDNet-2014 dataset with the following information. 
* Frames (From [changedetection.net](https://www.changedetection.net/))
* Ground truths (From [changedetection.net](https://www.changedetection.net/))
* Pre-computed recent background frames (Median of previous 30 frames)
* Pre-selected empty background frames
* Foreground probability maps (FPM) for all inputs (We used [DeepLab v3](https://github.com/tensorflow/models/tree/master/research/deeplab) for computing FPMs)

This dataset can be used test the functions in this repository. Upon publication, we will publicly share the full dataset with all of these inputs.

## Training the model
`train.py` can be used for training BSUV-Net 2.0. For the usage o arguments run `python train.py -h`

Training of BSUV-Net 2.0 on test_dataset: 
`python train.py  --set_number 5`

Training of Fast BSUV-Net 2.0 on test_dataset:
`python train.py  --seg_ch 0 --set_number 5`

These codes will save the perfomance of trained algorithm on the test dataset to `log_cv.csv`

## Cross-Validation Results


## Visualization