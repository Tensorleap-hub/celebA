from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', classes=[0., 1.], y=y_true[:, i]).astype(np.float32)
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        return K.mean \
            ((weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
             axis=-1)

    return weighted_loss


def model_loss(y_true, y_pred):
    weights = calculating_class_weights(y_true)
    y_true = tf.cast(y_true, tf.float32)
    return K.mean \
        ((weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
         axis=-1)
