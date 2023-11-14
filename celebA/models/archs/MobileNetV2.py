from keras.models import Model
from keras.layers import Dropout, Dense, BatchNormalization
from keras.applications.mobilenet_v2 import MobileNetV2

from celebA.config import IMAGE_SIZE, LABELS


def build_model(num_features: int = LABELS):
    base = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                       weights="imagenet",
                       include_top=False,
                       pooling='avg')  # GlobalAveragePooling 2D

    # models top
    x = base.output
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    top = Dense(num_features, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=top)



