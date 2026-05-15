# GTAvatar-Efficient-Garment-Transfer-for-Aligned-Avatars


This repository contains the implementation of the CAGD 2026 submission 
[GTAvatar: Efficient Garment Transfer for Aligned Avatars Simultaneously Reconstructed from Monocular Videos].

You can find detailed usage instructions for using pretrained models and training your own models below.

If you find our code useful, please cite:

```bibtex
@article
{fang2026gtavatar,
title={GTAvatar: Efficient garment transfer for aligned avatars simultaneously reconstructed from monocular videos},
author={Fang, Xianyong and Li, Jiarui and Zhou, Baofeng and Wang, Linbo and Liu, Zhengyi},
journal={Computer Aided Geometric Design},
pages={102566},
year={2026},
publisher={Elsevier}}
```

## Installation
### Environment Setup
This repository has been tested on the following platform:
1) Python 3.9.19, PyTorch 2.0.1 with CUDA 11.8 and cuDNN 8.7.0, Ubuntu 20.04

To clone the repo, run either:
```
git clone --recursive https://github.com/1815565/GTAvatar-Efficient-Garment-Transfer-for-Aligned-Avatars.git
```
or
```
git clone https://github.com/1815565/GTAvatar-Efficient-Garment-Transfer-for-Aligned-Avatars.git
cd GTAvatar-Efficient-Garment-Transfer-for-Aligned-Avatars
git submodule update --init --recursive
```

Next, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `gtavatar` using
```
conda env create -f environment.yml
conda activate gtavatar
# install tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Dataset preparation
Please follow the instructions of [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE) to preprocess the datasets.
Please follow the instructions of [Instant-NVR](https://github.com/zju3dv/instant-nvr) to to run SCHP (Self-Correction for Human Parsing) in order to get semantic files.

## Training
To train new networks from scratch, first, modify the "subjects" in "main/config", then run
```shell
python train.py --subject_id $exp_name 
```
To train MCA, we mainly follow the code from [GaussianIP](https://github.com/silence-tang/GaussianIP), but we did modify some files which are provided in GaussianIP folder.

## Evaluation
To evaluate the method , run
```shell
python test.py --subject_id $exp_name --test_epoch 5
```

## Acknowledgement
This project is built on source codes from [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE), [GaussianIP](https://github.com/silence-tang/GaussianIP). 
We sincerely thank these authors for their awesome work.
