# ADNI Multimodal MRI Research Codebase

This repository contains notebooks, preprocessing utilities, evaluation helpers, and 3D CNN model definitions used for ADNI-related MRI / segmentation / multimodal classification experiments.

## Structure

- `fastsurfer_stats_feature_preprocessing.py`
  - Extracts morphology features from FastSurfer `aseg+DKT.stats` files and saves one `.npy` file per subject/session.
- `evaluation_scripts/evaluate.py`
  - Computes common binary-classification metrics (F1, balanced accuracy, AUROC, AUPRC, specificity, sensitivity).
- `files_for_MedicalNet/`
  - 3D ResNet model definitions adapted for:
    - single-channel input (`resnet_cls.py`)
    - stacked two-channel input (`resnet_stack_cls.py`)
    - dual-branch fusion by addition (`resnet_stack_fusion_concat.py`)
- `*.ipynb`
  - Experiment notebooks for training and testing unimodal and multimodal models.

## Requirements

The codebase primarily uses:

- Python 3
- `numpy`
- `pandas`
- `tqdm`
- `scikit-learn`
- `torch` (for model definitions and training notebooks)

Notebook-specific dependencies may also include image-processing and neuroimaging libraries depending on the experiment workflow.

## Usage

### 1) FastSurfer morphology feature extraction

```bash
python3 fastsurfer_stats_feature_preprocessing.py \
  --dataset-csv path/to/dataset.csv \
  --stats-path-template '/path/to/fastsurfer/{subject_id}_{session_id}/stats/aseg+DKT.stats' \
  --output-dir path/to/output_features
```

Optional arguments:

- `--subject-col`: subject ID column name (default: first CSV column)
- `--session-col`: session ID column name (default: second CSV column)
- `--max-rows`: process only the first N rows (useful for debugging)
- `--expected-length`: expected feature length sanity check (default: `700`)
- `--strict`: stop on the first file error instead of skipping invalid rows

### 2) Evaluate prediction results

```bash
python3 evaluation_scripts/evaluate.py \
  --results-csv path/to/test_results.csv
```

Optional arguments:

- `--label-col`: label column name (default: `diagonsis`, matching existing notebooks)
- `--pred-col`: predicted label column (default: second-to-last column)
- `--prob-col`: predicted probability column (default: last column)

## Notebooks

The notebooks contain experiment pipelines for:

- morphology-feature-only experiments
- unimodal MRI / segmentation training
- multimodal stacked-input training
- fusion-based multimodal training

Before running a notebook, review the dataset paths and environment-specific settings in the first few code cells.

