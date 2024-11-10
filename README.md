# packaged InstantSplat

This repo is the **refactored python training and inference code for [InstantSplat](https://github.com/NVlabs/InstantSplat)**.
Forked from commit [2c5006d41894d06464da53d5495300860f432872](https://github.com/NVlabs/InstantSplat/tree/2c5006d41894d06464da53d5495300860f432872).
We **refactored the original code following the standard Python package structure**, while **keeping the algorithms used in the code identical to the original version**.
## Install

### Requirements

Install Pytorch and torchvision following the official guideline: [pytorch.org](https://pytorch.org/)

Install Pillow, numpy and tqdm
```sh
pip install Pillow numpy tqdm
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

### Local Install

```shell
git clone https://github.com/yindaheng98/InstantSplat --recursive
cd InstantSplat
pip install --target . --upgrade .
```

### Pip Install

```shell
pip install --upgrade git+https://github.com/yindaheng98/InstantSplat.git@main
```

## Running

1. Initialize coarse point cloud and jointly train 3DGS & cameras
```shell
python train.py -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --init
```

2. Render it
```shell
python render.py -s data/sora/santorini/3_views -d output/sora/santorini/3_views -i 1000 --load_camera output/sora/santorini/3_views/cameras.json
```

(Optional) Initialize coarse point and save as a Colmap workspace
```shell
python initialize.py -d data/sora/santorini/3_views
```

## Usage

**See [initialize.py](initialize.py), [train.py](train.py) and [render.py](render.py) for full example.**

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
