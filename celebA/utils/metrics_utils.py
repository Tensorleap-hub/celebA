from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric
from keras import backend

from celebA.config import LABELS


def calculate_binary_metrics(y_true, y_pred) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Calculate TP, TN, FP, FN for each label in a multi-label binary classification task using TensorFlow.

    Args:
        y_true (tf.Tensor): True labels tensor (0 or 1) for each label.
        y_pred (tf.Tensor): Predicted labels tensor (0 or 1) for each label.

    Returns:
        tp (tf.Tensor): True Positives for each label.
        tn (tf.Tensor): True Negatives for each label.
        fp (tf.Tensor): False Positives for each label.
        fn (tf.Tensor): False Negatives for each label.
    """

    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    tp = tf.cast(tf.logical_and(y_true, y_pred), tf.float32)
    tn = tf.cast(tf.logical_and(~y_true, ~y_pred), tf.float32)
    fp = tf.cast(tf.logical_and(~y_true, y_pred), tf.float32)
    fn = tf.cast(tf.logical_and(y_true, ~y_pred), tf.float32)

    return tp, tn, fp, fn


def class_accuracy(y_true, y_pred, cls_ind) -> tf.Tensor:
    """
    Calculate Accuracy metric per given class.

    Args:
        y_true (tf.Tensor): True labels tensor (0 or 1) for each label.
        y_pred (tf.Tensor): Predicted labels tensor (0 or 1) for each label.
        cls_ind: the class index

    Returns:
        tp (tf.Tensor): accuracy score.
    """
    y_pred = y_pred[:, cls_ind:cls_ind + 1]
    y_true = y_true[:, cls_ind:cls_ind + 1]
    return backend.mean(tf.equal(y_true, y_pred), axis=-1)


@tensorleap_custom_metric('calc_class_metrics_dic')
def calc_class_metrics_dic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate multiple metrics for each class.

    Args:
        y_true (tf.Tensor): True labels tensor (0 or 1) for each label.
        y_pred (tf.Tensor): Predicted probabilities tensor for each label.
        decision threshold

    Returns:
        dictionary with multi metrics scores
    """
    threshold = 0.5
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)

    res_dic = dict()
    tps, tns, fps, fns = calculate_binary_metrics(y_true, y_pred)
    for cls in LABELS:
        cls_ind = LABELS.index(cls)

        acc = class_accuracy(y_true, y_pred, cls_ind)
        out = tf.reduce_sum(y_pred[:, cls_ind:cls_ind + 1], -1)
        tp = tf.reduce_sum(tps[:, cls_ind:cls_ind + 1], -1)
        tn = tf.reduce_sum(tns[:, cls_ind:cls_ind + 1], -1)
        fp = tf.reduce_sum(fps[:, cls_ind:cls_ind + 1], -1)
        fn = tf.reduce_sum(fns[:, cls_ind:cls_ind + 1], -1)

        # Convert all to numpy arrays
        res_dic[f"{cls}_out"] = out.numpy()
        res_dic[f"{cls}_acc"] = acc.numpy()
        res_dic[f"{cls}_tp"] = tp.numpy()
        res_dic[f"{cls}_tn"] = tn.numpy()
        res_dic[f"{cls}_fp"] = fp.numpy()
        res_dic[f"{cls}_fn"] = fn.numpy()
    return res_dic
