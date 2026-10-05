# packaged InstantSplat

[![PyPI version](https://img.shields.io/pypi/v/InstantSplat.svg?logo=pypi)](https://pypi.org/project/InstantSplat/)
[![Downloads](https://api.pepy.tech/personalized-badge/InstantSplat?period=month&left_color=grey&right_color=brightgreen&left_text=monthly%20downloads)](https://pepy.tech/project/InstantSplat)
[![Total downloads](https://api.pepy.tech/personalized-badge/InstantSplat?period=total&left_color=grey&right_color=brightgreen&left_text=total%20downloads)](https://pepy.tech/project/InstantSplat)
[![CI](https://github.com/yindaheng98/InstantSplat/actions/workflows/build-release-linux.yml/badge.svg)](https://github.com/yindaheng98/InstantSplat/actions/workflows/ci.yml)
[![CI](https://github.com/yindaheng98/InstantSplat/actions/workflows/build-release-win.yml/badge.svg)](https://github.com/yindaheng98/InstantSplat/actions/workflows/ci.yml)

This repo is the **refactored python training and inference code for [InstantSplat](https://github.com/NVlabs/InstantSplat)**.
Forked from commit [2c5006d41894d06464da53d5495300860f432872](https://github.com/NVlabs/InstantSplat/tree/2c5006d41894d06464da53d5495300860f432872).
We **refactored the original code following the standard Python package structure**, while **keeping the algorithms used in the code identical to the original version**.

Initialization methods:
- [x] DUST3R (same method used in [InstantSplat](https://github.com/NVlabs/InstantSplat))
- [x] MAST3R (same method used in [Splatt3R](https://github.com/btsmart/splatt3r))
- [x] COLMAP Sparse reconstruct (same method used in [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting))
- [x] COLMAP Dense reconstruct (use `patch_match_stereo`, `stereo_fusion`, `poisson_mesher` and `delaunay_mesher` in COLMAP to reconstruct dense point cloud for initialization)
- [x] Masking during COLMAP feature extraction and dense fusion. See [File layout](#file-layout).
- [x] VGGT
- [x] [VGG-T³](https://github.com/nv-dvl/vgg-ttt)
- [ ] VGGT/VGG-T³ + Colmap Bundle Adjustment with COLMAP needs point tracking implemented on [`AbstractViewPointTracker`](https://github.com/yindaheng98/track-4dgs) from [track-4dgs](https://github.com/yindaheng98/track-4dgs).
- [x] Map-Anything and Map-Anything with external pose/depth priors

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)

Install a colmap executable, e.g. using conda:
```sh
conda install conda-forge::colmap
```

Install `mapanything` and `vggt`:
```shell
pip install --upgrade "mapanything[all] @ git+https://github.com/facebookresearch/map-anything.git@main"
pip install --upgrade git+https://github.com/facebookresearch/vggt.git@main
pip install --upgrade Pillow hydra-core omegaconf # deps for vggt
pip install --upgrade git+https://github.com/jytime/LightGlue.git#egg=lightglue # deps for vggt
```

(Optional) Install `xformers` for faster Depth-Anything V2 inference:
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
pip install --target . --upgrade --no-deps . --no-build-isolation
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
wget -P checkpoints/ https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/vggt_1B_commercial.pt --header="Authorization: Bearer $HF_TOKEN"
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth
wget -P checkpoints/ https://download.europe.naverlabs.com/ComputerVision/Pow3R/Pow3R_ViTLarge_BaseDecoder_512_linear.pth
```

Configs for `map-anything`:
```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/map-anything.git /tmp/map-anything-configs
git -C /tmp/map-anything-configs sparse-checkout set configs
cp -r /tmp/map-anything-configs/configs ./
rm -rf /tmp/map-anything-configs
```

## File layout

`<name>.jpg` is the image path relative to the image folder, including its extension and any subdirectory. `images/foo/bar.jpg` pairs with `image_masks/foo/bar.jpg.png`, `depths/foo/bar.jpg.tiff`, and `depth_masks/foo/bar.jpg.tiff`. Lists are collected with `os.walk`. Training reads the written scene as a [gaussian-splatting dataset](https://github.com/yindaheng98/gaussian-splatting#dataset-format).

### Input: `images/`

`dust3r`, `mast3r`, `mapanything`, `mapanything-external`, `vggt`, `vggttt`:

```
<data>/
  images/
    <name>.jpg
  image_masks/                 # optional
    <name>.jpg.png
```

### Input: `input/`

`colmap-sparse`, `colmap-dense`, `dust3r-align-colmap-sparse`, `dust3r-align-colmap-dense`:

```
<data>/
  input/
    <name>.jpg
  feature_mask/                # optional; COLMAP feature extraction only
    <name>.jpg.png
  input_mask/                  # optional; undistorted into image_masks/
    <name>.jpg.png
```

`feature_mask` and `input_mask` are independent. A missing mask is skipped.

### Output: no depth

`dust3r`, `mast3r`, `colmap-sparse`, `dust3r-align-colmap-sparse`:

```
<data>/
  images/
    <name>.jpg
  image_masks/                 # only when an input mask was provided
    <name>.jpg.png
  sparse/0/
    cameras.bin                # cameras.txt is also accepted when training
    images.bin
    points3D.ply               # if missing, points3D.bin then points3D.txt
```

### Output: depth

`vggt`, `vggttt`, `mapanything`, `mapanything-external`, `colmap-dense`, `dust3r-align-colmap-dense`. Same as above, plus:

```
<data>/
  depths/
    <name>.jpg.tiff
    <name>.jpg.png             # uint8 preview; the loader uses the tiff when both exist
  depth_masks/
    <name>.jpg.tiff
```

`--with_depth_anything` adds these depth files for any initializer. That depth is inverse depth. The initializers in this section otherwise write regular depth. See [Running](#running).

## Running

1. Initialize coarse point cloud and jointly train 3DGS & cameras
```shell
# Option 1: init and train in one command
python -m instantsplat.train -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --init dust3r
# Option 2: init and train in two separate commands
python -m instantsplat.initialize -d data/sora/santorini/3_views -i dust3r # init coarse point and save as a Colmap workspace at data/sora/santorini/3_views
python -m instantsplat.train -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 # train
```

To enable the optional auto-scaled Depth-Anything V2 wrapper for any initializer, add `--with_depth_anything`:
```shell
python -m instantsplat.initialize -d data/sora/santorini/3_views -i vggt --with_depth_anything
python -m instantsplat.train -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --init mapanything --with_depth_anything
```

Depth format note:
- `Depth-Anything V2` saves inverse depth (`1 / depth`), which matches the default depth supervision used by 3DGS.
- The native depth saved by `mapanything`, `mapanything-external`, `vggt`, `vggttt`, and COLMAP dense reconstruction is regular depth, not inverse depth.
- When training from those native depth maps without `--with_depth_anything`, add `-o depth_ground_truth_is_inversed=False`.

Example:
```shell
python -m instantsplat.train -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --init vggt -o depth_ground_truth_is_inversed=False
python -m instantsplat.train -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --init mapanything-external -o depth_ground_truth_is_inversed=False
```

### VGG-T³ initialization

`vggttt` initializes cameras, a point cloud, and native depth maps. Put images in `<scene>/images`.

The model weights are downloaded automatically from [`nvidia/vgg-ttt`](https://huggingface.co/nvidia/vgg-ttt) on first use. For a source checkout, clone with `--recursive` as shown in [Development Install](#development-install) so that the `submodules/vgg-ttt` submodule is available.

```shell
python -m instantsplat.initialize -d data/my_scene -i vggttt
```

The VGG-T³ inference options can be passed with `-o`:
```shell
python -m instantsplat.initialize -d data/my_scene -i vggttt \
  -o num_ttt_steps=2 \
  -o memory_efficient_inference=True \
  -o use_global_pred=True
```

VGG-T³ saves regular depth rather than inverse depth. When training without `--with_depth_anything`, use:
```shell
python -m instantsplat.train -s data/my_scene -d output/my_scene -i 1000 \
  --init vggttt -o depth_ground_truth_is_inversed=False
```

VGG-T³ code and weights are subject to the [NVIDIA OneWay Noncommercial License](https://github.com/nv-dvl/vgg-ttt/blob/main/LICENSE).

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
image_path_list = sorted(
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(image_folder)
    for filename in filenames
)
initializer = Dust3rInitializer(...).to(args.device) # see instantsplat/initializer/dust3r/dust3r.py for full options
initialized_point_cloud, initialized_cameras = initializer(image_path_list, destination)
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
