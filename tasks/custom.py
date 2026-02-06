import numpy as np

import datautils
from .classification import eval_classification


def eval_custom_classification(
    model,
    data,
    labels,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    axis=0,
    shuffle=True,
    seed=1234,
    encoding_protocol='full_series',
    eval_protocol='linear',
):
    """
        Evaluate a model on a custom dataset by splitting arrays directly.

    Returns:
        tuple: (y_score, eval_res, splits)
    """
    splits = datautils.split_timeseries_array(
        data,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        axis=axis,
        shuffle=shuffle,
        seed=seed,
    )
    train_data, val_data, test_data, train_labels, val_labels, test_labels = splits

    y_score, eval_res = eval_classification(
        model,
        train_data,
        train_labels,
        test_data,
        test_labels,
        encoding_protocol=encoding_protocol,
        eval_protocol=eval_protocol,
    )
    return y_score, eval_res, {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
    }
