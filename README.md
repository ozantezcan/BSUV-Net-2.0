# BSUV-Net-2.0

## Requirements
1. [Python 3.6.9](https://www.python.org/)
2. [PyTorch 1.3](https://pytorch.org/)
3. [OpenCV 4.0.1](https://opencv.org/releases/)

## Dataset
This repository includes a sampled version of CDNet-2014 dataset with the following: 
* Frames (From [changedetection.net](http://changedetection.net/))
* Ground truths (From [changedetection.net](http://changedetection.net/))
* Pre-computed recent background frames (Median of previous 30 frames)
* Pre-selected empty background frames
* Foreground probability maps (FPM) for all inputs (We used [DeepLab v3](https://github.com/tensorflow/models/tree/master/research/deeplab) for computing FPMs)

This dataset can be used to test the functions in this repository. Upon publication, we will publicly share the full dataset with all of these inputs.

## Training the mode
`train.py` can be used for training BSUV-Net 2.0. Run `python train.py -h` for seeing the usage of arguments

Training of BSUV-Net 2.0 on test_dataset: 
`python train.py  --set_number 5`

Training of Fast BSUV-Net 2.0 on test_dataset:
`python train.py  --seg_ch 0 --set_number 5`

These codes will save the performance of the trained algorithm on the test split specified in `configs/full_cv_config.py` to `log.csv`

## Cross-Validation
In `log.cv`, we provide the performance results for all 4 folds of the proposed cross-validation on the full dataset with BSUV-Net-2.0.
Follow the steps in `notebooks/crossvalidation.ipynb` to analyze locally-computed cross-validation results.

## Visualization of Spatio-Temporal Data Augmentations
Follow the steps in `notebooks/visualization.ipynb` to visualize spatio-temporal data augmentations.
