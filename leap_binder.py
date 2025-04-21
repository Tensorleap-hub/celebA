from typing import Union

import pandas as pd
import numpy as np
import PIL.Image as Image
from keras import backend as K
import tensorflow as tf

# Tensorleap imports
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader import leap_binder

from celebA.utils.gcs_utils import _download
from celebA.data.preprocess import preprocess_response
from celebA.config import *
from celebA.utils.metrics_utils import calc_class_metrics_dic
from celebA.training import class_weights


# Input encoder fetches the image with the index `idx` from the data from set in
# the PreprocessResponse's data.
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    tf_data = preprocess.data['tf_data']
    sample = next(iter(tf_data.skip(idx)))

    fname = sample[0].numpy().decode('utf-8')
    fpath = image_path + f'/{fname}'
    fpath = _download(fpath)
    image = Image.open(fpath)

    # center crop
    width, height = image.size
    left = (width - celeba_face_size) / 2
    top = (height - celeba_face_size) / 2
    right = (width + celeba_face_size) / 2
    bottom = (height + celeba_face_size) / 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    return np.array(image) / 255


def get_sample_row(idx: int, preprocess: Union[PreprocessResponse, list]) -> pd.Series:
    tf_data = preprocess.data['tf_data']
    cols = preprocess.data['columns']
    sample = next(iter(tf_data.skip(idx)))
    labels_values = sample[1].numpy().astype(np.int32)
    return pd.Series(index=cols, data=labels_values[:len(cols)])


def gt_encoder(idx: int, preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
    row = get_sample_row(idx, preprocess)
    labels_vec = np.array(row[LABELS] == 1)
    return labels_vec.astype(np.float32)


def metadata_dic_vals(idx: int, preprocess: Union[PreprocessResponse, list]) -> dict:
    row = get_sample_row(idx, preprocess).astype(np.float32)
    row = dict(row)
    for k, v in row.items():
        row[k] = float(v)
    return dict(row)


def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    """ Use the default TL horizontal bar just with the classes names added """
    return LeapHorizontalBar(np.squeeze(data), LABELS)


def model_weighted_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    weights = class_weights()
    return (K.mean
            ((weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
         axis=-1)).numpy()


# -------------- Dataset binding functions: --------------


leap_binder.set_preprocess(function=preprocess_response)

leap_binder.set_input(function=input_encoder, name='image')

leap_binder.set_ground_truth(function=gt_encoder, name='classes')

leap_binder.add_prediction(name='classes', labels=LABELS)

leap_binder.set_metadata(metadata_dic_vals, 'metadata_dic')

leap_binder.add_custom_metric(calc_class_metrics_dic, 'class_metrics_dic')

leap_binder.set_visualizer(name='horizontal_bar_classes', function=bar_visualizer,
                           visualizer_type=LeapHorizontalBar.type)

leap_binder.add_custom_loss(name='weighted_loss', function=model_weighted_loss)

if __name__ == "__main__":
    leap_binder.check()
