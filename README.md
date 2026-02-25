# YOLO-IOD: Towards Real Time Incremental Object Detection

Official Pytorch implementation for "*YOLO-IOD: Towards Real Time Incremental Object Detection*", AAAI 2026 Poster. [[Paper](https://arxiv.org/abs/2512.22973)]

![image-20250818142604338](./assets/framework.png)

## 🚀 Contributions

- We introduce YOLO-IOD, an integrated and real-time IOD framework, and pinpoint three causes of forgetting: foreground-background confusion, parameter interference, and misaligned knowledge distillation conflict.
- YOLO-IOD incorporates three innovative modules aimed at mitigating forgetting, with particular emphasis on the dual-teacher CAKD module. This module addresses the challenge of misaligned knowledge distillation by channeling the target student detector’s features through the detection heads of both the former and current teacher detectors.
- We introduce LoCo COCO, a practical benchmark designed to remove image overlap between stages and consider category co-occurrence, allowing for a more equitable assessment of incremental object detection.

## Get Started

This repo is based on [YOLO-World](https://github.com/AILab-CVC/YOLO-World). Please follow the installation of YOLO-World and make sure you can run it successfully.

```
conda create -n yolo-iod python=3.10.16 -y
conda activate yolo-iod
# Please follow the official YOLO-World Installation Guide to set up this environment (PyTorch, MMCV, MMEngine, etc.).
cd our project
pip install -v -e .
cd third_party/mmyolo/
cd our project
```

## Data Preparation

**Organize the dataset structure:**

```
data/
└── coco/
    │── train2017/
    │── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

**setting split**

```
python utils/coco2017_split.py
```

## Pre-trained Models

YOLO-IOD is constructed upon the pre-trained YOLO-World model. You will need to download the appropriate pre-trained YOLO-World weights. Please refer to the [YOLO-World official repository](https://github.com/xxxxx/YOLO-World) .

Place the pre-trained weights `x_stage1-62b674ad.pth` in a designated `weights/` directory:

## Unknown Pseudo label

```
python utils/pseudo_unknown_label.py
```

## Train

```
bash run_40+40.sh
```

