import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset, normalize=True):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    if normalize:
        scaler = StandardScaler()
        scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
        train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
        test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens


def load_forecast_csv(name, univar=False, raw=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    if not raw:
        scaler = StandardScaler().fit(data[train_slice])
        data = scaler.transform(data)
    else:
        scaler = None

    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
    
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data


def load_npz_array(path, key=None):
    """
        Load a custom (N, T, C) numpy array from a .npz file.

    Args:
        path (str): Path to the .npz file on disk.
        key (str, optional): Name of the array to load. If None, tries 'data'
            or uses the single array found in the file.

    Returns:
        np.ndarray: Array with shape (N, T, C).
    """
    with np.load(path) as data:
        if key is None:
            if 'data' in data:
                key = 'data'
            elif len(data.files) == 1:
                key = data.files[0]
            else:
                raise ValueError(
                    "NPZ contains multiple arrays; specify which one to load via 'key'."
                )
        array = data[key]

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array with shape (N, T, C). Got shape {array.shape}.")
    return array


def save_npz_array(path, array, key='data'):
    """
        Save a numpy array to a .npz file.

    Args:
        path (str): Output path for the .npz file.
        array (np.ndarray): Array to save.
        key (str): Name of the array inside the .npz archive.
    """
    np.savez(path, **{key: array})


def save_encoded_with_labels(path, representations, labels, repr_key='reprs', label_key='labels'):
    """
        Save encoded representations alongside labels to a .npz file.

    Args:
        path (str): Output path for the .npz file.
        representations (np.ndarray): Encoded representations array.
        labels (np.ndarray): Labels corresponding to the representations.
        repr_key (str): Name of the representations array inside the archive.
        label_key (str): Name of the labels array inside the archive.
    """
    np.savez(path, **{repr_key: representations, label_key: labels})


def validate_timeseries_array(array, name='data'):
    """
        Validate that an array is a 3D time-series tensor.

    Args:
        array (np.ndarray): Array to validate.
        name (str): Name used in error messages.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, T, C). Got shape {array.shape}.")


def summarize_timeseries_array(array):
    """
        Summarize basic statistics for a (N, T, C) array.

    Returns:
        dict: Summary containing shape, missing ratio, and per-channel stats.
    """
    validate_timeseries_array(array)
    nan_mask = np.isnan(array)
    missing_ratio = nan_mask.mean()
    per_channel = {}
    for channel in range(array.shape[2]):
        channel_data = array[:, :, channel]
        per_channel[channel] = {
            'mean': np.nanmean(channel_data),
            'std': np.nanstd(channel_data),
            'min': np.nanmin(channel_data),
            'max': np.nanmax(channel_data),
        }
    return {
        'shape': array.shape,
        'missing_ratio': missing_ratio,
        'per_channel': per_channel,
    }


def split_timeseries_array(
    array,
    labels=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    axis=0,
    shuffle=True,
    seed=1234,
):
    """
        Split a time-series array (and optional labels) into train/val/test splits.

    Args:
        array (np.ndarray): Array to split.
        labels (np.ndarray, optional): Labels aligned with the split axis.
        train_ratio (float): Fraction for the training split.
        val_ratio (float): Fraction for the validation split.
        test_ratio (float): Fraction for the test split.
        axis (int): Axis to split along (0 for instances, 1 for time).
        shuffle (bool): Whether to shuffle before splitting (only when axis=0).
        seed (int): Random seed used for shuffling.

    Returns:
        tuple: (train, val, test) or (train, val, test, train_labels, val_labels, test_labels)
    """
    validate_timeseries_array(array)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (instances) or 1 (time).")

    length = array.shape[axis]
    indices = np.arange(length)

    if axis == 0 and shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_end = int(length * train_ratio)
    val_end = train_end + int(length * val_ratio)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if axis == 0:
        train = array[train_idx]
        val = array[val_idx]
        test = array[test_idx]
    else:
        train = array[:, train_idx]
        val = array[:, val_idx]
        test = array[:, test_idx]

    if labels is None:
        return train, val, test

    labels = np.asarray(labels)
    if labels.shape[0] != length:
        raise ValueError("Labels must align with the split axis length.")
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    return train, val, test, train_labels, val_labels, test_labels
