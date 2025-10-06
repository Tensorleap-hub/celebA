from typing import Tuple, List
import tensorflow as tf
import pandas as pd

# Tensorleap imports
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess

from celebA.config import att_path, partition_path, landmarks_path, align_landmarks_path, celeb_id_path, \
    train_size, val_size, test_size
from celebA.utils.gcs_utils import _download


def load_data():
    annotations_path = _download(att_path)
    df_attr = pd.read_csv(annotations_path, index_col=0)
    df_attr = df_attr.replace({-1: 0})
    local_landmarks_path = _download(landmarks_path)
    df_landmarks = pd.read_csv(local_landmarks_path, index_col=0)
    df_attr = df_attr.join(df_landmarks)

    local_landmarks_path = _download(align_landmarks_path)
    df_landmarks = pd.read_csv(local_landmarks_path, index_col=0)
    df_landmarks.columns = [col + '_align' for col in df_landmarks.columns]
    df_attr = df_attr.join(df_landmarks)

    local_celeb_id_path = _download(celeb_id_path)
    df_celeb_id = pd.read_csv(local_celeb_id_path, names=['0', 'celeb_id'], delimiter=' ').set_index('0')
    df_attr = df_attr.join(df_celeb_id)
    return df_attr


def split_by_partition(df_attr):
    local_partition_path = _download(partition_path)
    df_partition = pd.read_csv(local_partition_path, index_col=0)
    df_attr = df_attr.join(df_partition)

    df_train = df_attr[df_attr.partition == 0]
    df_valid = df_attr[df_attr.partition == 1]
    df_test = df_attr[df_attr.partition == 2]

    return df_train, df_valid, df_test


def df_to_tf_dataset(df_train, df_valid, df_test):
    tf_train = tf.data.Dataset.from_tensor_slices((df_train.index.values, df_train.values))
    tf_valid = tf.data.Dataset.from_tensor_slices((df_valid.index.values, df_valid.values))
    tf_test = tf.data.Dataset.from_tensor_slices((df_test.index.values, df_test.values))
    return tf_train, tf_valid, tf_test


def preprocess_data():
    df_atrr = load_data()
    cols = df_atrr.columns
    df_train, df_valid, df_test = split_by_partition(df_atrr)  # split subsets
    tf_train = tf.data.Dataset.from_tensor_slices((df_train.index.values, df_train.values))
    tf_valid = tf.data.Dataset.from_tensor_slices((df_valid.index.values, df_valid.values))
    tf_test = tf.data.Dataset.from_tensor_slices((df_test.index.values, df_test.values))

    return tf_train, tf_valid, tf_test, cols


# TL Preprocess Function:
@tensorleap_preprocess()
def preprocess_response() -> List[PreprocessResponse]:
    tf_train, tf_valid, tf_test, cols = preprocess_data()

    train = PreprocessResponse(length=train_size, data=dict(tf_data=tf_train, columns=cols))
    val = PreprocessResponse(length=val_size, data=dict(tf_data=tf_valid, columns=cols))
    test = PreprocessResponse(length=test_size, data=dict(tf_data=tf_test, columns=cols))
    return [train, val, test]
