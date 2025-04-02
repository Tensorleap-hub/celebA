from celebA.data.preprocess import preprocess_response
from leap_binder import input_encoder, gt_encoder, get_sample_row, metadata_dic_vals, calc_class_metrics_dic, model_weighted_loss
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

if __name__ == "__main__":

    subsets = preprocess_response()
    train, val, test = subsets
    i = 0
    set = train
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'celebA/models/GenderAgeBCELoss.h5'
    model = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    img = input_encoder(i, set)
    gt = gt_encoder(i, set)
    pred = model(np.expand_dims(img, 0))

    row = get_sample_row(i, set)
    res = calc_class_metrics_dic(y_true=np.expand_dims(gt, 0), y_pred=pred)
    loss = model_weighted_loss(y_true=np.expand_dims(gt, 0), y_pred=pred.numpy())

    res_batch = calc_class_metrics_dic(y_true=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), y_pred=np.array([[0.9, 0.3], [0.7, 0.2], [0.8, 0.99]]))
    loss_batch = model_weighted_loss(y_true=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), y_pred=np.array([[0.9, 0.3], [0.7, 0.2], [0.8, 0.99]]))
    metadata_vals = metadata_dic_vals(i, set)
    plt.imshow(img)
