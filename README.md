# T-Rep for Neuroscience Time-Series

T-Rep is a representation-learning model for time-series using time embeddings, with an interface that fits naturally into neuroscience workflows (EEG, ECoG, LFP, calcium imaging traces, spike-rate sequences). This repository contains the official implementation for ["T-Rep: Representation Learning for Time-Series Using Time-Embeddings"](https://arxiv.org/abs/2310.04486), with optional S3 input-level preprocessing for robust segment shuffling.

## Who this is for

If you are working with trial-based or continuous neural recordings and want a single encoder that can power decoding, clustering, forecasting, or anomaly detection, this project is for you. Typical use cases include:

- Trial-level classification (e.g., stimulus decoding, behavioral state inference)
- Event detection (e.g., onset detection, transient discovery)
- Representation learning for downstream regressors or classifiers
- Forecasting or anomaly detection on continuous recordings

## Data expectations

T-Rep expects `np.ndarray` inputs with shape `(N, T, C)`:

- `N`: number of trials, sessions, or instances
- `T`: number of time steps
- `C`: number of channels (e.g., sensors, regions, neurons)

Missing data should be represented as `NaN` values. If you have irregular sampling or gaps, pad with `NaN` to align sequences.

Example: if you have trial-aligned ECoG with 80 trials, 2000 time steps, and 64 channels, your array should have shape `(80, 2000, 64)`.

## Install

```bash
pip install -r requirements.txt
```

The repository is not yet compatible with PyTorch 2.0. We recommend a dedicated virtual environment with the pinned dependency versions.

## Neuroscience Quickstart

Below is a minimal end-to-end example. Customize preprocessing based on your modality.

```python
import numpy as np
import torch

from trep import TRep

# Example: load an .npz with your neural data stored as "data"
data = np.load("my_neuro_dataset.npz")["data"]  # shape (N, T, C)

# Optional preprocessing (example ideas):
# - bandpass filtering (EEG/ECoG)
# - z-score per channel
data = (data - np.nanmean(data, axis=(0, 1), keepdims=True)) / (
    np.nanstd(data, axis=(0, 1), keepdims=True) + 1e-6
)

model = TRep(
    input_dims=data.shape[-1],
    output_dims=128,
    time_embedding="t2v_sin",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

loss_log = model.fit(data, n_epochs=50, verbose=1)
reprs = model.encode(data)  # shape (N, T, output_dims)
```

## Optional S3 front-end (input-level shuffling)

S3 is an input-level module that stochastically reorders segments to encourage invariances to temporal jitter and local artifacts. For neural data with transient noise or timing jitter, S3 can act as a robustifying front-end.

```python
model = TRep(
    input_dims=data.shape[-1],
    output_dims=128,
    time_embedding="t2v_sin",
    use_s3=True,
    s3_layers=2,  # 1–3 layers recommended
    s3_initial_num_segments=4,
)
```

## Task guidance for neuroscience data

- **Trial classification / decoding**: encode representations per trial (e.g., mean pool over time or use a temporal classifier on `reprs`).
- **Event detection / anomaly detection**: use fine-grained, time-step representations and apply a per-time-step detector.
- **Forecasting**: train on earlier time spans and encode later spans with correct time indices to avoid time-embedding drift.

If your train/test split is time-based, preserve the original time indices to keep time embeddings consistent.

## CLI usage (for reproduction)

To train and evaluate on supported benchmark datasets:

```bash
python train.py <dataset_name> <run_name> --loader <loader> --repr-dims <repr_dims> --eval
```

Supported loaders include `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, and `anomaly_coldstart`. For a full list of arguments, run `python train.py -h`.

## Optional dependencies for neuroscience workflows

These packages are not required by T-Rep but are commonly useful:

- `mne`: EEG/MEG preprocessing and filtering
- `scipy`: signal processing utilities
- `h5py`: working with `.h5` datasets

## Reproducing paper results (benchmark datasets)

If you want to reproduce the original paper results, the benchmark datasets can be downloaded as follows:

- [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) → `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`
- [30 UEA datasets](http://www.timeseriesclassification.com) → `datasets/UEA/<dataset_name>/<dataset_name>_*.arff`
- [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) → `datasets/ETTh1.csv`, `datasets/ETTh2.csv`, `datasets/ETTm1.csv`
- [Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70) → preprocess with `datasets/preprocess_yahoo.py`
- [Sepsis dataset](https://physionet.org/content/challenge-2019/1.0.0/training/#files-panel) → preprocess with `datasets/preprocess_sepsis.py`

Evaluation scripts are in:

- `evaluation.py` for classification, forecasting, anomaly detection
- `clustering.py` for clustering
- `sepsis_ad.py` for sepsis anomaly detection

## Citation

If this work is useful in your research, please cite:

```bibtex
@inproceedings{
    fraikin2024trep,
    title={T-Rep: Representation Learning for Time Series using Time-Embeddings},
    author={Archibald Felix Fraikin and Adrien Bennetot and Stephanie Allassonniere},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=3y2TfP966N}
}
```

