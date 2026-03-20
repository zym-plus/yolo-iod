# Reproduce On Server

This project can now be reproduced on a Linux server with the helper scripts in `scripts_safe/`.
By default, Hugging Face downloads are routed through `https://hf-mirror.com`.
You can override the default before running scripts:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
```

The scripts and Python entry points print `[HF Mirror]` logs with the active
endpoint and cache paths. If downloads still fail, check those logs first. If
you still see `huggingface.co` requests, then some path is bypassing the mirror
initialization.

## 1. Environment

Create a Python 3.10 environment and install a PyTorch build that matches your CUDA driver first.

Example for CUDA 11.8:

```bash
conda create -n yolo-iod python=3.10 -y
conda activate yolo-iod
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
bash scripts_safe/install_repro.sh
```

Default cache environment:

- `HF_HOME=$HOME/.cache/huggingface`
- `HF_HUB_CACHE=$HF_HOME/hub`
- `HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE`
- `TRANSFORMERS_CACHE=$HF_HUB_CACHE`

## 2. Dataset Layout

Place the original COCO2017 dataset under `data/coco`:

```text
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

`scripts_safe/prepare_repro.sh` will:

- create `weights/` and `work_dirs/`
- download `weights/x_stage1-62b674ad.pth` if it is missing
- create the missing alias file `data/texts/unknown_class_texts.json`
- generate `data/coco/annotations/40+40(order)/...`
- generate `data/coco/loco_annotations/40+40(order)/...`

If the server cannot access the internet, prepare offline assets first on a
machine with internet:

```bash
bash scripts_safe/download_offline_assets.sh
```

This creates:

- `weights/x_stage1-62b674ad.pth`
- `pretrained_models/clip-vit-base-patch32/`

## 3. Run Experiments

Single GPU:

```bash
GPUS=1 bash scripts_safe/run_coco_40_40_full.sh
GPUS=1 bash scripts_safe/run_loco_40_40_full.sh
```

Multi GPU:

```bash
GPUS=4 bash scripts_safe/run_coco_40_40_full.sh
GPUS=4 bash scripts_safe/run_loco_40_40_full.sh
```

## 4. Notes

- If `pretrained_models/clip-vit-base-patch32/` exists, the project will use it directly and will not need to download CLIP from Hugging Face during runtime.
- If you already placed `weights/x_stage1-62b674ad.pth` manually, `prepare_repro.sh` will reuse it.
- You can force regeneration of split files with `FORCE_REGEN_SPLITS=1 bash scripts_safe/prepare_repro.sh`.
