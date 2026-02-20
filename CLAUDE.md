# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

M3D-NCA is a PyTorch-based framework for medical image segmentation using Neural Cellular Automata. It implements:
- **Med-NCA**: 2D multi-level segmentation
- **M3D-NCA**: 3D multi-level segmentation with variance-based quality control
- **Med-GrapherNCA**: ViG-style Grapher modules integrated into NCA for graph-based perception

Med-GrapherNCA defines two new model variants:
- **GrapherNCA_M1 (Pixel-Grapher-NCA)**: Each pixel is a graph node. KNN graph in feature space + max-relative aggregation **replaces** Sobel/conv perception entirely. Best at downsampled levels due to O(N^2) KNN.
- **GrapherNCA_M2 (Patch-Grapher-NCA)**: Image divided into patches, each patch = graph node. Grapher output is **concatenated with** standard conv perception (hybrid local+global).

Multi-level combinations pair two models (downsampled + full-res level): b1b1, m1b1, m1m1, m2b1, m2m2, m1m2.

Models are extremely lightweight (~13k parameters, 50kB storage). Training uses patch-based multi-level hierarchical learning to handle high-resolution medical images within VRAM constraints.

## Setup

```bash
pip install -r M3D-NCA/requirements.txt
```

Key dependencies: PyTorch, TorchIO (0.18.82), nibabel, opencv-python, tensorboard, seaborn, ipython.

For local development, use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchio==0.18.82 nibabel opencv-python tensorboard seaborn ipython
```

## Running Training

Training is driven through Jupyter notebooks at the repo root. Configure `img_path`, `label_path`, and `model_path` in the config dict before running.

### Original Med-NCA / M3D-NCA
- `M3D-NCA/train_Med_NCA.ipynb`, `M3D-NCA/train_M3D_NCA.ipynb`
- Dataset: Medical Decathlon NIfTI format (.nii.gz)

### Grapher-NCA (Google Colab)
- `train_GrapherNCA_single.ipynb` — single-level b1/m1/m2 experiments
- `train_GrapherNCA_multi.ipynb` — multi-level combinations (b1b1, m1b1, m1m1, m2b1, m2m2, m1m2)
- `evaluate_GrapherNCA.ipynb` — evaluation with Dice, IoU, pseudo-ensemble variance, visualization
- Dataset: ISIC 2018 skin lesion segmentation (JPEG/PNG, auto-downloaded in notebooks)
- Notebooks mount Google Drive at `/content/drive/MyDrive/Experiments/Grapher_NCA/` and download ISIC 2018 to `datasets/ISIC2018/`
- Code must be at `Experiments/Grapher_NCA/M3D-NCA/` in Drive (or update `REPO_DIR`)

```bash
# View training metrics
tensorboard --logdir <model_path>/tensorboard
```

There is no formal test suite or linting configuration. Evaluation is done via:
```python
agent.getAverageDiceScore(pseudo_ensemble=True, showResults=True)
```

## Architecture

### Training Flow

1. **Experiment** (`src/utils/Experiment.py`) — manages config, model paths, tensorboard writer, epoch state, and reload
2. **Dataset** (`src/datasets/`) — loads images, handles normalization, label binarization; `Nii_Gz_Dataset_3D` for NIfTI, `ISIC_Dataset` for ISIC 2018
3. **Model** (`src/models/`) — NCA cell definitions with perceive/update cycle
4. **Agent** (`src/agents/`) — orchestrates training loop, loss computation, evaluation, and checkpointing

### Agent Hierarchy

`BaseAgent` → `Agent_NCA` → `Agent_Multi_NCA` → `Agent_Med_NCA` (2D) / `Agent_M3D_NCA` (3D)

- **BaseAgent** (`Agent.py`): Core training loop, model save/load, TensorBoard logging
- **Agent_NCA**: NCA seed creation, pool management
- **Agent_Multi_NCA**: Multi-model batch step coordination
- **Agent_Med_NCA / Agent_M3D_NCA**: Multi-level inference with pooling/upsampling, patch-based training, pseudo-ensemble quality metrics

### NCA Cell Design

- **BasicNCA** (`Model_BasicNCA.py`): Sobel filter perception (dx, dy + identity), FC update layers, stochastic fire rate
- **BasicNCA3D** (`Model_BasicNCA3D.py`): Conv3d perception, batch normalization, stochastic updates on 5D tensors
- **BackboneNCA** (`Model_BackboneNCA.py`): Enhanced 2D with learnable Conv2d perception (b1 baseline)
- **GrapherNCA_M1** (`Model_GrapherNCA_M1.py`): Pixel-level graph perception replacing Sobel entirely. Uses `Grapher.py` modules (KNN, MRConv2d, GrapherModule, FFN). Each pixel = 1 node, reshapes [B,H,W,C] → [B,N,C] for graph ops.
- **GrapherNCA_M2** (`Model_GrapherNCA_M2.py`): Patch-level graph perception concatenated with Conv2d perception. Avg-pools to patches → graph ops → upsamples back. FC input is 4*channel_n (3C conv + C graph).

### Grapher Module (`Grapher.py`)

Shared graph infrastructure for m1 and m2:
- `pairwise_distance(x)` / `knn(x, k)` — KNN graph construction in feature space
- `MRConv2d` — max-relative graph convolution: `[x_i ; max(x_j - x_i)]` → Linear+BN+GELU
- `GrapherModule` — fc1 → KNN → MRConv → fc2 → residual
- `FFN` — Linear(C→4C)+BN+GELU → Linear(4C→C)+BN → residual

### Multi-Level Training

Models use a list of NCA cells, one per resolution level. Lower levels operate on downsampled images; higher levels use patches from the full resolution. Outputs are upsampled/combined across levels.

### ISIC 2018 Dataset (`ISIC_Dataset.py`)

- Loads JPEG images + PNG segmentation masks from directory structure
- `self.slice = 0` (non-None) signals 2D mode to the agent — **critical**: the agent uses `dataset.slice != None` to choose 2D vs 3D code paths
- Image filenames: `ISIC_XXXXXXX.jpg`, mask filenames: `ISIC_XXXXXXX_segmentation.png`
- Normalizes to [0,1], binarizes masks, supports RGB or grayscale via `input_channels` param

### Configuration

All hyperparameters are passed as Python dicts (no schema validation). Key fields: `input_size` (list of tuples per level), `inference_steps` (per level), `channel_n`, `cell_fire_rate`, `scale_factor`, `data_split`.

Grapher-specific config fields: `k` (KNN neighbors, default 9), `patch_size` (for m2, default 4).

### Loss Functions (`LossFunctions.py`)

DiceLoss, DiceBCELoss, BCELoss, FocalLoss, DiceFocalLoss, **IoULoss** (new — used in evaluation).

### Persistence

- `config.dt` — JSON-pickled config
- `data_split.dt` — pickled train/val/test split
- `model.pth`, `optimizer.pth`, `scheduler.pth` — PyTorch state dicts (multi-level: `model0.pth`, `model1.pth`, etc.)

### Key Conventions

- **Tensor format**: NCA models use channel-last `[B, H, W, C]` internally. Conv operations transpose to `[B, C, H, W]` and back.
- **Input channels frozen**: `forward()` preserves the first `input_channels` dimensions unchanged across NCA steps.
- **Stochastic fire rate**: Random per-cell mask applied to updates during both training and inference (enables pseudo-ensemble evaluation).
