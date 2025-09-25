# DADA + Hybrid Augmentation (TensorFlow)

This repository contains a TensorFlow implementation of DADA (Deep Adversarial Data Augmentation) and optional hybrid augmentations (Mixup, CutMix) for low-data image classification.

## Features
- DADA generator–discriminator training with class-conditional generation.
- Optional hybrid augmentation window applied to real images (Mixup/CutMix).
- Clean TF `tf.data` pipelines for CIFAR-10 / SVHN.
- Reproducible configs via environment variables or `config.py`.
- Lightweight TensorBoard logging.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Minimal requirements.txt:

```
tensorflow>=2.12
numpy
scikit-learn
matplotlib
```

## Quick start
```bash
# Default config (dataset=cifar10)
python train.py

# Example: override via env
DADA_DATASET=cifar10 DADA_BATCH_SIZE=128 DADA_LR=3e-4 python train.py
```

## Configuration

Configuration lives in `dada_tf/config.py` (exported from the original `dada_config.py`). You can:

- Edit defaults in `config.py`, or
- Override via environment variables, e.g.:

  - `DADA_DATASET` (cifar10 or svhn)
  - `DADA_BATCH_SIZE`, `DADA_LR`, `DADA_G_LR`, `DADA_D_LR`
  - Scheduling variables if present (see `utils/schedules.py`)
  - Hybrid augmentation window variables if present (start/end epochs, alpha, etc.)

Run `python train.py --help` if an argparse layer exists; otherwise read `config.py` for available keys.

## Project layout
```
dada_tf/
  augmentations.py      # Mixup/CutMix
  generator.py          # DADA generator (32x32 outputs)
  discriminator.py      # Class-conditional D with 2*C logits
  losses.py             # DADA loss computations
  weightnorm_wrapper.py # Weight normalization utilities
  config.py             # Training configuration (from env or defaults)
  train.py              # Orchestrates data/model/training/eval
  utils/
    data.py             # Dataset loading, normalization, subsampling, tf.data, image grids
    schedules.py        # LR schedule helpers
    metrics.py          # Accuracy/F1 + confusion matrix
    tfops.py            # Small tensor helpers shared across modules
    logging.py          # TensorBoard/file logging helpers
```

## Reproducibility

Set seeds in `config.py`.

Determinism may vary across GPUs and TF versions; compare short-run metrics, not exact bitwise weights.

## Logging
```
tensorboard --logdir logs
```


