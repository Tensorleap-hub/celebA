from typing import Tuple, Dict, Any

import numpy as np
from code_loader.contract.enums import MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric

from celebA.config import LABELS


def calculate_binary_metrics(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate TP, TN, FP, FN for each label in a multi-label binary classification task using NumPy.

    Args:
        y_true (np.ndarray): True labels array (0 or 1) for each label.
        y_pred (np.ndarray): Predicted labels array (0 or 1) for each label.

    Returns:
        tp (np.ndarray): True Positives for each label.
        tn (np.ndarray): True Negatives for each label.
        fp (np.ndarray): False Positives for each label.
        fn (np.ndarray): False Negatives for each label.
    """

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    # Boolean masks for presence of positives/negatives for each class (dim: num_classes,)
    has_pos = np.any(y_true, axis=0)
    has_neg = np.any(~y_true, axis=0)

    # Initialize outputs with NaNs
    tp = np.full_like(y_true, np.nan, dtype=np.float32)
    tn = np.full_like(y_true, np.nan, dtype=np.float32)
    fp = np.full_like(y_true, np.nan, dtype=np.float32)
    fn = np.full_like(y_true, np.nan, dtype=np.float32)

    # Only compute for classes where true positives or negatives exist
    tp[:, has_pos] = (y_true[:, has_pos] & y_pred[:, has_pos]).astype(np.float32)
    fn[:, has_pos] = (y_true[:, has_pos] & ~y_pred[:, has_pos]).astype(np.float32)
    tn[:, has_neg] = (~y_true[:, has_neg] & ~y_pred[:, has_neg]).astype(np.float32)
    fp[:, has_neg] = (~y_true[:, has_neg] & y_pred[:, has_neg]).astype(np.float32)

    return tp, tn, fp, fn


def class_accuracy(y_true, y_pred, cls_ind) -> np.array:
    """
    Calculate Accuracy metric per given class.

    Args:
        y_true (np.array): True labels tensor (0 or 1) for each label.
        y_pred (np.array): Predicted labels tensor (0 or 1) for each label.
        cls_ind: the class index

    Returns:
        tp (np.array): accuracy score.
    """
    y_pred = y_pred[:, cls_ind:cls_ind + 1]
    y_true = y_true[:, cls_ind:cls_ind + 1]
    return np.mean(y_true == y_pred, axis=-1)


@tensorleap_custom_metric('calc_class_metrics_dic',
                          compute_insights= {
                            **{f'{cls}_out': False for cls in LABELS},
                            **{f'{cls}_acc': True for cls in LABELS},
                            **{f'{cls}_tp': True for cls in LABELS},
                            **{f'{cls}_tn': True for cls in LABELS},
                            **{f'{cls}_fp': True for cls in LABELS},
                            **{f'{cls}_fn': True for cls in LABELS}},
                          direction={
                              **{f'{cls}_acc': MetricDirection.Upward for cls in LABELS},
                              **{f'{cls}_tp': MetricDirection.Upward for cls in LABELS},
                              **{f'{cls}_tn': MetricDirection.Upward for cls in LABELS},
                              **{f'{cls}_fp': MetricDirection.Downward for cls in LABELS},
                              **{f'{cls}_fn': MetricDirection.Downward for cls in LABELS}})
def calc_class_metrics_dic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate multiple metrics for each class.

    Args:
        y_true (np.ndarray): True labels tensor (0 or 1) for each label.
        y_pred (np.ndarray): Predicted probabilities tensor for each label.
        decision threshold

    Returns:
        dictionary with multi metrics scores
    """
    threshold = 0.5
    y_pred = y_pred > threshold

    res_dic = dict()
    tps, tns, fps, fns = calculate_binary_metrics(y_true, y_pred)
    for cls in LABELS:
        cls_ind = LABELS.index(cls)

        acc = class_accuracy(y_true, y_pred, cls_ind)
        out = np.sum(y_pred[:, cls_ind:cls_ind + 1], -1)
        tp = np.sum(tps[:, cls_ind:cls_ind + 1], -1)
        tn = np.sum(tns[:, cls_ind:cls_ind + 1], -1)
        fp = np.sum(fps[:, cls_ind:cls_ind + 1], -1)
        fn = np.sum(fns[:, cls_ind:cls_ind + 1], -1)

        res_dic[f"{cls}_out"] = out
        res_dic[f"{cls}_acc"] = acc
        res_dic[f"{cls}_tp"] = tp
        res_dic[f"{cls}_tn"] = tn
        res_dic[f"{cls}_fp"] = fp
        res_dic[f"{cls}_fn"] = fn
    return res_dic
