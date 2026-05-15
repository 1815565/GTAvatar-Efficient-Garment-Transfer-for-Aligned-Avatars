# GTAvatar-Efficient-Garment-Transfer-for-Aligned-Avatars

<img src="assets/teaser.gif" width="800"/> 

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
1) Python 3.7.13, PyTorch 1.12.1 with CUDA 11.6 and cuDNN 8.3.2, Ubuntu 22.04/CentOS 7.9.2009

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
To train new networks from scratch, run
```shell
# ZJU-MoCap
python train.py dataset=zjumocap_377_mono
# PeopleSnapshot
python train.py dataset=ps_female_3 option=iter30k pose_correction=none 
```
To train on a different subject, simply choose from the configs in `configs/dataset/`.

We use [wandb](https://wandb.ai) for online logging, which is free of charge but needs online registration.

## Evaluation
To evaluate the method for a specified subject, run
```shell
# ZJU-MoCap
python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono
# PeopleSnapshot
python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ps_female_3
```

## Test on out-of-distribution poses
First, please download the preprocessed AIST++ and AMASS sequence for subjects in ZJU-MoCap [here](https://drive.google.com/drive/folders/17vGpq6XGa7YYQKU4O1pI4jCMbcEXJjOI?usp=drive_link) 
and extract under the corresponding subject folder `${ZJU_ROOT}/CoreView_${SUBJECT}`.

To animate the subject under out-of-distribution poses, run
```shell
python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

We provide four preprocessed sequences for each subject of ZJU-MoCap, 
which can be specified by setting `dataset.predict_seq` to 0,1,2,3, 
where `dataset.predict_seq=3` corresponds to the canonical rendering.

Currently, the code only supports animating ZJU-MoCap models for out-of-distribution models.

## License
We employ [MIT License](LICENSE) for the 3DGS-Avatar code, which covers
```
configs
dataset
models
utils/dataset_utils.py
extract_smpl_parameters.py
render.py
train.py
```

The rest of the code are modified from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 
Please consult their license and cite them.

## Acknowledgement
This project is built on source codes from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 
We also use the data preprocessing script and part of the network implementations from [ARAH](https://github.com/taconite/arah-release).
We sincerely thank these authors for their awesome work.
