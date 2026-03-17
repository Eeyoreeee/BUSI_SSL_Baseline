# BUSI SSL Baseline

Clean, small-scope PyTorch baseline scaffold for BUSI breast ultrasound binary lesion segmentation.

This repository is being built from scratch for a first-stage baseline only. It is not a paper-author code cleanup, not a trainer migration, and not a platform-style framework.

## Current Status

The repository is currently aligned through Step 5.

Step 1 is completed:

- project skeleton
- pinned dependencies
- minimal YAML config system with one-level base inheritance
- training entrypoint skeletons and data script skeletons

Step 2 is completed:

- `tools/prepare_busi_manifest.py`
- `tools/make_splits.py`
- manifest helpers
- BUSI supervised dataset
- SSL dataset
- transforms
- dataloaders

Step 3 is completed:

- DeepLabV3Plus model wrapper
- supervised loss
- evaluator
- checkpoint manager
- logger
- `SupervisedTrainer`
- runnable `train_sup.py`

Step 4 is completed for the minimal SSL path:

- EMA teacher
- fixed-threshold pseudo-labeling
- lambda_u warmup
- masked unlabeled BCE loss
- minimal `SSLTrainer`
- runnable `train_ssl.py`

Step 5 is completed for detached analysis logging:

- independent `StatsCollector`
- epoch-level pseudo-label diagnostics
- analysis metrics written to `train.log`, `metrics.csv`, and optional `tb/`

Not implemented yet:

- GT-based debug analysis
- boundary / transition / prototype analysis
- analysis-driven training feedback

`train_sup.py` and `train_ssl.py` are both runnable once BUSI data preparation has been completed. The SSL path remains intentionally minimal, and analysis stays detached from all training decisions.

## Stage-1 Scope

Implemented target for stage 1:

- task: binary lesion segmentation
- dataset: BUSI
- default data usage: benign + malignant only, exclude normal
- merged lesion mask: offline OR merge before training
- input size: 512x512
- input channels: always 3
- model family: DeepLabV3Plus + ResNet50 only
- training lines:
  - supervised-only
  - minimal teacher-student SSL with EMA and fixed-threshold pseudo-labeling

Explicitly not part of stage 1:

- UNet
- dynamic threshold
- distribution alignment
- boundary supervision
- transition/core/hard-region logic
- prototype consistency
- fp branch
- CutMix
- complex distillation variants
- multi-teacher or multi-student
- PyTorch Lightning
- Hydra
- MONAI
- registry or hook platform
- any analysis-to-training feedback logic

## Environment

Required versions for this project:

- Python 3.10.14
- torch==2.4.0
- torchvision==0.19.0
- segmentation-models-pytorch==0.5.0

Additional minimal packages are pinned in `requirements.txt`.

Install example:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model Weight Semantics

The stage-1 model config is intentionally strict about weight wording.

When the config uses:

```yaml
encoder_weights: imagenet
```

the project will describe the encoder only as:

`standard ImageNet-pretrained ResNet50 encoder`

and not as:

- official torchvision ResNet50 weights
- equivalent to official torchvision ResNet50 weights

The config fields explicitly include:

```yaml
encoder_weight_source: segmentation_models_pytorch::resnet50/imagenet
encoder_weight_note: standard ImageNet-pretrained ResNet50 encoder; not claimed as official torchvision ResNet50 weights
claim_official_torchvision_equivalence: false
```

This repository does not claim strict equivalence to torchvision official ResNet50 weights when using `segmentation_models_pytorch`.

## Input And Preprocessing Contract

- model input is always 3-channel
- if a BUSI image is grayscale, it will be read as `H x W x 1` and repeated to `H x W x 3`
- stage 1 will not use `in_channels=1`
- spatial size is fixed to `512x512`
- normalization follows ImageNet mean/std semantics in the segmentation data flow
- current stage-1 and step-5 geometry augmentation is an engineering tradeoff:
  - first resize inputs to `512x512`
  - then apply shared geometry perturbations on that resized canvas
  - this is for stability and implementation clarity in the baseline, not a claim of universally optimal augmentation policy

## Data Preparation Contract

BUSI-specific cleanup is handled offline before training.

Generated artifacts:

- `data_meta/manifest.csv`
- `data_meta/merged_masks/<sample_id>.png`
- `data_meta/splits/seed0/*.txt`

Manifest columns:

- `sample_id`
- `label_group`
- `image_relpath`
- `merged_mask_relpath`
- `mask_count`
- `component_mask_relpaths`
- `image_sha1`

Rules:

- all manifest path fields are relative paths
- `sample_id` is deterministic and does not depend on row order
- split files store only `sample_id`
- multi-mask aggregation is completed offline before training
- `image_relpath` and `component_mask_relpaths` are relative to `dataset_root`
- `merged_mask_relpath` is always relative to `merged_mask_root`
- the current implementation stores `merged_mask_relpath` as `<sample_id>.png`, which is still just the simplest relative path under `merged_mask_root`
- `merged_mask_relpath` must not be interpreted as a sometimes-filename / sometimes-full-relative-path field
- default `sample_id` is `<label_group>__<normalized_stem>`
- if that base id collides, the suffix is a stable hash fragment from `image_sha1`
- runtime counters are not used for collision handling

Current split design:

- `train.txt`
- `val.txt`
- `test.txt`
- `labeled_subset.txt`
- `unlabeled_pool.txt`
- `overfit_8_samples.txt`
- `smoke_labeled.txt`
- `smoke_unlabeled.txt`

`unlabeled_pool.txt` is an explicit engineering choice so that SSL trainers do not need to compute set differences at runtime.

## Training Output Contract

Both supervised and minimal SSL training write the same fixed output layout:

```text
outputs/<exp_name>/
|-- config_dump.yaml
|-- train.log
|-- metrics.csv
|-- tb/
`-- checkpoints/
    |-- best.pth
    `-- last.pth
```

`best.pth` and `last.pth` always live under `checkpoints/`. TensorBoard logs go under `tb/`. The resolved config dump is always named `config_dump.yaml`.

When `analysis.enabled=true` for SSL training, epoch-level detached analysis metrics are appended to the same `metrics.csv` and optional TensorBoard stream. They remain diagnostics only.

## Config System Contract

- YAML only
- no Hydra
- only one-level inheritance: `child -> base`
- base configs cannot inherit another base
- merge rule is limited to shallow merging of known top-level sections
- no recursive multi-level config framework

Base configs:

- `configs/base_sup.yaml`
- `configs/base_ssl.yaml`

## Implemented Files

Step 1:

- `requirements.txt`
- `train_ssl.py`
- `busi_seg/config.py`
- `configs/base_sup.yaml`
- `configs/base_ssl.yaml`
- `configs/experiments/sup_subset.yaml`
- `configs/experiments/sup_full.yaml`
- `configs/experiments/ssl_tau090.yaml`
- `configs/experiments/ssl_tau095.yaml`
- `configs/experiments/ssl_tau0975.yaml`
- `configs/verify/overfit_8_samples.yaml`
- `configs/verify/ssl_smoke_test.yaml`

Step 2:

- `tools/prepare_busi_manifest.py`
- `tools/make_splits.py`
- `busi_seg/data/manifest.py`
- `busi_seg/data/busi_dataset.py`
- `busi_seg/data/ssl_dataset.py`
- `busi_seg/data/transforms.py`
- `busi_seg/data/loaders.py`

Step 3:

- `train_sup.py`
- `busi_seg/models/deeplabv3plus.py`
- `busi_seg/models/builder.py`
- `busi_seg/losses/dice.py`
- `busi_seg/losses/supervised_loss.py`
- `busi_seg/engine/evaluator.py`
- `busi_seg/engine/checkpoint.py`
- `busi_seg/logging/logger.py`
- `busi_seg/engine/trainer_sup.py`

Step 4:

- `busi_seg/ssl/ema.py`
- `busi_seg/ssl/pseudo_labeler.py`
- `busi_seg/ssl/ramps.py`
- `busi_seg/losses/masked_bce.py`
- `busi_seg/engine/trainer_ssl.py`
- `train_ssl.py`

Step 5:

- `busi_seg/analysis/stats_collector.py`

## What Comes Next

The next stage can expand diagnostics if needed, for example:

- GT-based debug analysis
- richer detached analysis views

Training feedback from analysis remains out of scope for this baseline stage.
