# ULIP Manage

## Pre-trained model

### ULIP2

```bash
https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt
```

### OpenCLIP

```bash
git lfs install
git clone https://ai.gitee.com/hf-models/CLIP-ViT-bigG-14-laion2B-39B-b160k --no-checkout
git lfs track "open_clip_pytorch_model.bin"
git lfs pull -I "open_clip_pytorch_model.bin"
```


## Setup

```bash
conda create -n ulip python=3.10
conda activate ulip
./setup.sh
```

## Run

```bash
python demo.py
```

## Enjoy it~
