# BSUV-Net-2.0

## Requirements
1. [Python 3.6.9](https://www.python.org/)
2. [PyTorch 1.3](https://pytorch.org/)
3. [OpenCV 4.0.1](https://opencv.org/releases/)
4. [tensorboardX 2.2](https://github.com/lanpa/tensorboardX)
5. [matplotlib](https://matplotlib.org/)

## Dataset
Steps for preparing CDNet2014
1. Download the dataset from [changedetection.net](http://changedetection.net/) and unzip the contents in `./dataset`

2. Download pre-computed empty and recent background frames with FPM images from [Google Drive](https://drive.google.com/drive/folders/1fskxV1paCsoZvqTVLjnlAdPOCHk1_XmF?usp=sharing) and place the contents in `./dataset`

3. In the end, `./dataset` folder should have the following subfolders: `currentFr`, `currentFrFpm`, `emptyBg`, `emptyBgFpm`, `recentBg`, `recentBgFpm`.

## Cross-validation

1. Run `python train.py  --set_number <k>` for `<k> = 1, 2, 3 and 4` to compute the results for each fold. This code will save the results to `log.csv`.

2. Follow the steps in `notebooks/crossvalidation.ipynb` to analyze cross-validation results.

## Visualization of Spatio-Temporal Data Augmentations
Follow the steps in `notebooks/visualization.ipynb` to visualize spatio-temporal data augmentations.

## Inference on unseen video
Use this repo for inference: [https://github.com/ozantezcan/BSUV-Net-inference](https://github.com/ozantezcan/BSUV-Net-inference)

## Training and Cross-Validation with other datasets.
Change `./configs/data_config.py` and `./configs/full_cv_config.py` for training BSUV-Net 2.0 with different datasets.

