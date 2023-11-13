from celebA.data.preprocess import preprocess_response
from leap_binder import input_encoder, gt_encoder, get_sample_row, metadata_dic_vals, calc_class_metrics_dic
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    subsets = preprocess_response()
    train, val, test = subsets
    i = 0
    res = calc_class_metrics_dic(y_true=np.array([[1, 1], [1, 1], [1, 1]]), y_pred=np.array([[0.9, 0.3], [0.7, 0.2], [0.8, 0.99]]))
    metadata_vals = metadata_dic_vals(i, train)
    img = input_encoder(i, train)
    gt = gt_encoder(i, train)
    row = get_sample_row(i, train)
    row = get_sample_row(i, val)
    metadata_vals = metadata_dic_vals(i, val)
    metadata_vals = metadata_dic_vals(i, test)
    plt.imshow(img)
