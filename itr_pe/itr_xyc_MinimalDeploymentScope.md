# itr_xyc Minimal Deployment Scope

This document summarizes the smallest file set needed to deploy the clothing
print extraction inference path based on `itr_xyc/IS-Net/Inference.py` and the
provided IS-Net weight file.

## Core Result

There are two deployment levels:

1. Mask inference only:
   - Runs `Inference.py`.
   - Produces one grayscale PNG mask per input image.
   - Minimum project payload: about 168.56 MiB.

2. End-to-end print cutout:
   - Runs `Inference.py`, then `apply_mask.py`.
   - Produces RGBA images by applying the predicted mask to the original image.
   - Minimum project payload is still about 168.56 MiB; `apply_mask.py` is only about 1.8 KiB.
   - Uses Pillow at runtime; OpenCV is not required in this local deployment package.

The weight file dominates the upload size:

```text
itr_xyc/gpu_itr_98000_traLoss_0.0577_traTarLoss_0.0027_valLoss_0.0726_valTarLoss_0.0047_maxF1_0.9521_mae_0.0024_time_0.027661.pth
```

Size: 176,720,018 bytes, about 168.53 MiB.

## Required Files

Upload these files for mask inference:

```text
itr_xyc/IS-Net/Inference.py
itr_xyc/IS-Net/models/__init__.py
itr_xyc/IS-Net/models/isnet.py
itr_xyc/gpu_itr_98000_traLoss_0.0577_traTarLoss_0.0027_valLoss_0.0726_valTarLoss_0.0047_maxF1_0.9521_mae_0.0024_time_0.027661.pth
itr_xyc/requirements-lj.txt
```

Add this file if you also need RGBA print cutouts:

```text
itr_xyc/IS-Net/apply_mask.py
```

## Suggested Server Layout

```text
itr_xyc_deploy/
  Inference.py
  apply_mask.py                  # optional
  requirements-lj.txt
  models/
    __init__.py
    isnet.py
  weights/
    gpu_itr_98000_traLoss_0.0577_traTarLoss_0.0027_valLoss_0.0726_valTarLoss_0.0047_maxF1_0.9521_mae_0.0024_time_0.027661.pth
  input/
  output_detect/
  masked/                        # optional
```

## Files Not Needed For Inference Deployment

These files/directories are not required for the minimal inference path:

```text
itr_xyc/IS-Net/IS-Net/
itr_xyc/IS-Net/__pycache__/
itr_xyc/IS-Net/models/__pycache__/
itr_xyc/IS-Net/new_models/
itr_xyc/IS-Net/basics.py
itr_xyc/IS-Net/data_loader_cache.py
itr_xyc/IS-Net/hce_metric_main.py
itr_xyc/IS-Net/my_train.py
itr_xyc/IS-Net/rename_split.py
itr_xyc/IS-Net/train_valid_inference_main.py
itr_xyc/IS-Net/pytorch18.yml
itr_xyc/IS-Net/requirements.txt
```

Root `itr_xyc/requirements.txt` from the original project is a Conda explicit
environment export, not a minimal deployment requirements file.

## Code Dependency Findings

- `Inference.py` imports `ISNetDIS` and supports both script execution and package import.
- `models/__init__.py` exports `ISNetDIS` and `ISNetGTEncoder` from `models/isnet.py` with package-import compatibility.
- The provided `.pth` loads cleanly into `ISNetDIS` with no missing or unexpected keys.
- The nested `itr_xyc/IS-Net/IS-Net/` copy is a duplicate of the top-level `IS-Net` code and can be skipped.
- `new_models/` is not used by `Inference.py` and is not compatible with the provided weight path unless code is changed.

## Local Deployment Edits Already Applied

The local `itr_pe` scripts now resolve paths relative to their own folder and
also provide command-line arguments:

- `Inference.py`
  - `--input-dir`
  - `--output-dir`
  - `--model-path`
  - `--device`
  - `--input-size`
- `apply_mask.py`
  - `--input-dir`
  - `--mask-dir`
  - `--output-dir`
  - `--threshold`

Output folders are created automatically.

## Minimal Runtime Dependencies

Use an environment with matching `torch` and `torchvision`. Then install only:

```bash
pip install -r requirements-lj.txt
```

The current local scripts do not require `scikit-image` or OpenCV.
