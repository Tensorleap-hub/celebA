from celebA.data.preprocess import preprocess_response
from leap_binder import input_encoder, gt_encoder, metadata_dic_vals, calc_class_metrics_dic, \
    model_weighted_loss, bar_visualizer
import numpy as np
import os
import tensorflow as tf

from leap_binder import leap_binder
from code_loader.helpers import visualize

if __name__ == "__main__":
    check_generic = True
    plot_vis = True

    if check_generic:
        leap_binder.check()

    print("Started custom test")
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

    #metrics
    res = calc_class_metrics_dic(y_true=np.expand_dims(gt, 0), y_pred=pred.numpy())
    loss = model_weighted_loss(y_true=np.expand_dims(gt, 0), y_pred=pred.numpy())

    res_batch = calc_class_metrics_dic(y_true=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), y_pred=np.array([[0.9, 0.3], [0.7, 0.2], [0.8, 0.99]]))
    loss_batch = model_weighted_loss(y_true=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), y_pred=np.array([[0.9, 0.3], [0.7, 0.2], [0.8, 0.99]]))

    #vis
    gt_vis = bar_visualizer(np.expand_dims(gt, 0))
    pred_vis = bar_visualizer(pred.numpy())

    if plot_vis:
        visualize(gt_vis)
        visualize(pred_vis)

    #metadata
    metadata_vals = metadata_dic_vals(i, set)


    print("Finish custom test")