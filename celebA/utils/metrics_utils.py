from typing import Tuple

import tensorflow as tf
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


def calc_class_metrics_dic(y_true, y_pred, threshold=0.5):
    """
    Calculate multiple metrics for each class.

    Args:
        y_true (tf.Tensor): True labels tensor (0 or 1) for each label.
        y_pred (tf.Tensor): Predicted probabilities tensor for each label.
        decision threshold

    Returns:
        dictionary with multi metrics scores
    """
    y_pred = tf.convert_to_tensor(y_pred)
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)

    res_dic = dict()
    tps, tns, fps, fns = calculate_binary_metrics(y_true, y_pred)
    for cls in LABELS:
        cls_ind = LABELS.index(cls)
        res = class_accuracy(y_true, y_pred, cls_ind)
        res_dic[f"{cls}_out"] = tf.cast(tf.squeeze(y_pred[:, cls_ind:cls_ind + 1]), tf.float32)
        res_dic[f"{cls}_acc"] = res
        res_dic[f"{cls}_tp"] = tf.reduce_sum(tps[:, cls_ind:cls_ind + 1], -1)
        res_dic[f"{cls}_tn"] = tf.reduce_sum(tns[:, cls_ind:cls_ind + 1], -1)
        res_dic[f"{cls}_fp"] = tf.reduce_sum(fps[:, cls_ind:cls_ind + 1], -1)
        res_dic[f"{cls}_fn"] = tf.reduce_sum(fns[:, cls_ind:cls_ind + 1], -1)
    return res_dic
