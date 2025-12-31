import pandas as pd
import PIL.Image as Image
from code_loader.default_metrics import categorical_crossentropy
from code_loader.visualizers.default_visualizers import default_image_visualizer
from code_loader.inner_leap_binder.leapbinder_decorators import *

# Tensorleap imports
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader import leap_binder

from celebA.utils.gcs_utils import _download
from celebA.config import *


@tensorleap_input_encoder('image', channel_dim = -1)
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

    return (np.array(image) / 255).astype(np.float32)


def get_sample_row(idx: int, preprocess: Union[PreprocessResponse, list]) -> pd.Series:
    tf_data = preprocess.data['tf_data']
    cols = preprocess.data['columns']
    sample = next(iter(tf_data.skip(idx)))
    labels_values = sample[1].numpy().astype(np.int32)
    return pd.Series(index=cols, data=labels_values[:len(cols)])


@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
    row = get_sample_row(idx, preprocess)
    labels_vec = np.array(row[LABELS] == 1)
    return labels_vec.astype(np.float32)


@tensorleap_metadata('metadata_dic')
def metadata_dic_vals(idx: int, preprocess: Union[PreprocessResponse, list]) -> dict:
    row = get_sample_row(idx, preprocess).astype(np.float32)
    row = dict(row)
    for k, v in row.items():
        row[k] = float(v)
    return dict(row)


@tensorleap_custom_visualizer('horizontal_bar_classes', LeapHorizontalBar.type)
def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    """ Use the default TL horizontal bar just with the classes names added """
    return LeapHorizontalBar(np.squeeze(data), LABELS)


@tensorleap_custom_visualizer('image_viz', LeapDataType.Image)
def image_viz(data):
    return default_image_visualizer(data)


@tensorleap_custom_loss('categorical_crossentropy_loss')
def categorical_crossentropy_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)


# -------------- Dataset binding functions: --------------


if __name__ == "__main__":
    leap_binder.check()
