# packaged InstantSplat

This repo is the **refactored python training and inference code for [InstantSplat](https://github.com/NVlabs/InstantSplat)**.
Forked from commit [2c5006d41894d06464da53d5495300860f432872](https://github.com/NVlabs/InstantSplat/tree/2c5006d41894d06464da53d5495300860f432872).
We **refactored the original code following the standard Python package structure**, while **keeping the algorithms used in the code identical to the original version**.

Initialization methods:
- [x] DUST3R (same method used in [InstantSplat](https://github.com/NVlabs/InstantSplat))
- [x] MAST3R (same method used in [Splatt3R](https://github.com/btsmart/splatt3r))
- [x] COLMAP Sparse reconstruct (same method used in [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting))
- [x] COLMAP Dense reconstruct (use `patch_match_stereo`, `stereo_fusion`, `poisson_mesher` and `delaunay_mesher` in COLMAP to reconstruct dense point cloud for initialization)
- [x] Masking of keypoints during COLMAP feature extraction (just put your mask into `mask` folder, e.g. for an image `data/xxx/input/012.jpg`, the mask would be `data/xxx/input_mask/012.jpg.png`)

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)

Install a colmap executable, e.g. using conda:
```sh
conda install conda-forge::colmap
```

(Optional) Install `xformers` for faster depth anything:
```shell
pip install xformers
```

(Optional) If you have trouble with [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting), try to install it from source:
```sh
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
```

## PyPI Install

```shell
pip install --upgrade instantsplat
```
or
build latest from source:
```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/InstantSplat.git@main --no-build-isolation
```

### Development Install

```shell
git clone --recursive https://github.com/yindaheng98/InstantSplat
cd InstantSplat
pip install scipy huggingface_hub einops roma scikit-learn
pip install --target . --upgrade --no-deps .
```

## Download model

```sh
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

## Running

1. Initialize coarse point cloud and jointly train 3DGS & cameras
```shell
# Option 1: init and train in one command
python -m instantsplat.train -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --init dust3r
# Option 2: init and train in two separate commands
python -m instantsplat.train -i dust3r -d data/sora/santorini/3_views -i dust3r # init coarse point and save as a Colmap workspace at data/sora/santorini/3_views
python -m instantsplat.train -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 # train
```

2. Render it
```shell
python -m gaussian_splatting.render -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --load_camera output/sora/santorini/3_views/cameras.json
```

See [.vscode\launch.json](.vscode\launch.json) for more command examples.

## Usage

**See [instantsplat.initialize](instantsplat/initialize.py), [instantsplat.train](instantsplat/train.py) and [gaussian_splatting.render](https://github.com/yindaheng98/gaussian-splatting/blob/master/gaussian_splatting/render.py) for full example.**

Also check [yindaheng98/gaussian-splatting](https://github.com/yindaheng98/gaussian-splatting) for more detail of training process.

### Gaussian models

Use `CameraTrainableGaussianModel` in [yindaheng98/gaussian-splatting](https://github.com/yindaheng98/gaussian-splatting)

### Dataset

Use `TrainableCameraDataset` in [yindaheng98/gaussian-splatting](https://github.com/yindaheng98/gaussian-splatting)

### Initialize coarse point cloud and cameras

```python
from instantsplat.initializer import Dust3rInitializer
image_path_list = [os.path.join(image_folder, file) for file in sorted(os.listdir(image_folder))]
initializer = Dust3rInitializer(...).to(args.device) # see instantsplat/initializer/dust3r/dust3r.py for full options
initialized_point_cloud, initialized_cameras = initializer(image_path_list=image_path_list)
```

Create camera dataset from initialized cameras:
```python
from instantsplat.initializer import TrainableInitializedCameraDataset
dataset = TrainableInitializedCameraDataset(initialized_cameras).to(device)
```

Initialize 3DGS from initialized coarse point cloud:
```python
gaussians.create_from_pcd(initialized_point_cloud.points, initialized_point_cloud.colors)
```

### Training

`Trainer` jointly optimize the 3DGS parameters and cameras, without densification
```python
from instantsplat.trainer import Trainer
trainer = Trainer(
    gaussians,
    dataset=dataset,
    ... # see instantsplat/trainer/trainer.py for full options
)
```

<h2 align="center"> <a href="https://arxiv.org/abs/2403.20309">InstantSplat: Sparse-view SfM-free <a href="https://arxiv.org/abs/2403.20309"> Gaussian Splatting in Seconds </a>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.20309) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kairunwen/InstantSplat) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://instantsplat.github.io/)[![X](https://img.shields.io/badge/-Twitter@Zhiwen%20Fan%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/WayneINR/status/1774625288434995219)  [![youtube](https://img.shields.io/badge/Demo_Video-E33122?logo=Youtube)](https://youtu.be/fxf_ypd7eD8) [![youtube](https://img.shields.io/badge/Tutorial_Video-E33122?logo=Youtube)](https://www.youtube.com/watch?v=JdfrG89iPOA&t=347s)
</h5>

<div align="center">
This repository is the official implementation of InstantSplat, an sparse-view, SfM-free framework for large-scale scene reconstruction method using Gaussian Splatting.
InstantSplat supports 3D-GS, 2D-GS, and Mip-Splatting.
</div>
<br>

## Free-view Rendering
https://github.com/zhiwenfan/zhiwenfan.github.io/assets/34684115/748ae0de-8186-477a-bab3-3bed80362ad7

## TODO List
- [ ] Confidence-aware Point Cloud Downsampling
- [ ] Support 2D-GS
- [ ] Support Mip-Splatting

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [DUSt3R](https://github.com/naver/dust3r)

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@misc{fan2024instantsplat,
        title={InstantSplat: Unbounded Sparse-view Pose-free Gaussian Splatting in 40 Seconds},
        author={Zhiwen Fan and Wenyan Cong and Kairun Wen and Kevin Wang and Jian Zhang and Xinghao Ding and Danfei Xu and Boris Ivanovic and Marco Pavone and Georgios Pavlakos and Zhangyang Wang and Yue Wang},
        year={2024},
        eprint={2403.20309},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
      }
```
