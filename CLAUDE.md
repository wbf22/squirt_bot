# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`squirt_bot` is a deep learning model that identifies targets from camera input and triggers a liquid sprayer. It performs few-shot segmentation: given positive ("do") and negative ("don't") example images, it produces a binary mask over an input frame indicating where the target is.

## Setup

```bash
bash setup.sh          # creates venv and installs torch/torchvision/torchaudio
source venv/bin/activate
```

## Running

There is no training script or inference runner yet — only the model definition in `model.py`. Run experiments directly:

```bash
python model.py
```

## Architecture (`model.py`)

`TargetDetector` is a single `nn.Module` with three stages:

1. **CNN backbone** (`convolutions` method) — ResNet-style encoder with 6 conv blocks and 7 residual blocks. Extracts two feature pyramid levels:
   - `p3`: 256-channel intermediate features (after `conv_3`)
   - `p5`: deep features reduced to `EMBED_DIM_HALF` (128) via `conv_6`
   - Both are projected to `EMBED_DIM_HALF` and concatenated on the channel dim to form a 256-d (`EMBED_DIM`) feature map.

2. **Cross-attention** — `MultiheadAttention` where:
   - **Query**: feature map of the input image (flattened spatial patches)
   - **Key/Value**: example embeddings concatenated with the semantic difference (`examples − anti_examples`), encoding "what to spray" minus "what to ignore"

3. **Mask decoder** (not yet implemented) — intended to upsample attention output back to input resolution and apply sigmoid for a binary segmentation mask.

**Loss**: BCE + Dice loss (BCE handles per-pixel accuracy; Dice handles class imbalance since most pixels are background).

**Key constants**: `EMBED_DIM=256`, `KERNAL_SIZE=3`, `STRIDE=2`, `PADDING=1`.

**`forward` signature**:
```python
forward(input_image,    # [B, 3, H, W]
        examples,       # [B, num_examples*H'*W', EMBED_DIM]  — positive example embeddings
        anti_examples)  # [B, num_examples*H'*W', EMBED_DIM]  — negative example embeddings
```
Example images should be pre-processed through `convolutions()` before being passed to `forward`.
