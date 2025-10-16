import os

from code_loader.contract.datasetclasses import PreprocessResponse, PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.plot_functions.visualize import visualize

from celebA.config import LABELS
from celebA.data.preprocess import preprocess_response
from celebA.utils.metrics_utils import calc_class_metrics_dic
from leap_binder import input_encoder, gt_encoder, categorical_crossentropy_loss, bar_visualizer, metadata_dic_vals, \
    image_viz


@tensorleap_load_model([PredictionTypeHandler('classes', LABELS)])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'celebA/models/GenderAgeBCELoss.h5'

    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    return model


@tensorleap_integration_test()
def check_custom_integration(idx, preprocess_subset: PreprocessResponse):
    print("started custom tests")

    model = load_model()
    img = input_encoder(idx, preprocess_subset)
    gt = gt_encoder(idx, preprocess_subset)
    pred = model(img)

    res = calc_class_metrics_dic(y_true=gt, y_pred=pred)

    loss = categorical_crossentropy_loss(y_true=gt, y_pred=pred)

    # vis
    gt_vis = bar_visualizer(gt)
    pred_vis = bar_visualizer(pred)
    input_viz = image_viz(img)

    visualize(gt_vis)
    visualize(pred_vis)
    visualize(input_viz)

    metadata_vals = metadata_dic_vals(idx, preprocess_subset)

    print("Finish custom test")


if __name__ == "__main__":
    responses = preprocess_response()
    train = responses[0]
    check_custom_integration(0, train)
