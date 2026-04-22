# TST: Typical-Smoothed Truncation for Out-of-Distribution Detection

This repository contains the official implementation of **TST (Typical-Smoothed Truncation)** for out-of-distribution detection.

## Overview

TST transforms fixed-threshold truncation into a bidirectional smoothing filter to improve OOD detection performance. The method:
1. Normalizes the channel-aware typical set
2. Smooths and corrects abnormal features through a bidirectional low-pass filter
3. Computes the energy score for the corrected features

## Usage

### 1. Dataset Preparation

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/id_data/ILSVRC-2012/train` and `./datasets/id_data/ILSVRC-2012/val`, respectively.

For CIFAR-100 experiments, the dataset will be automatically downloaded.

#### Out-of-distribution dataset

We use 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf).

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, download from the [original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Place all OOD datasets into `./datasets/ood_data/`.

### 2. Pre-trained Model Preparation

The pre-trained models (ResNet-50, ResNet-18, MobileNet-V2) are provided by PyTorch and will be downloaded automatically.

### 3. OOD Detection Evaluation

#### ImageNet-1K evaluation:
```bash
python eval-resnet.py --in-dataset imagenet --name mobilenet --method energy
```

#### CIFAR-100 evaluation:
```bash
python eval-cifar.py --in-dataset CIFAR-100 --name wrn --method energy
```

## Project Structure

```
TST-main/
├── eval-cifar.py          # CIFAR-100 evaluation script
├── eval-resnet.py         # ImageNet evaluation script (TST method)
├── eval_itp_imagenet.py   # ITP baseline evaluation
├── eval_tsre.py           # TSRE baseline evaluation
├── getfeat.py             # Feature extraction utility
├── score.py               # OOD scoring functions
├── models/
│   ├── mobilenet.py       # MobileNet-V2 model
│   ├── resnet.py          # ResNet models
│   └── wrn.py             # Wide ResNet model
└── util/
    ├── args_loader.py     # Argument parsing
    ├── data_loader.py     # Data loading utilities
    ├── metrics.py         # Evaluation metrics (FPR95, AUROC)
    └── model_loader.py    # Model loading utilities
```

## Acknowledgment

This code is built upon [ReAct: Out-of-distribution Detection With Rectified Activations](https://openreview.net/pdf?id=IBVBtz_sRSm).
