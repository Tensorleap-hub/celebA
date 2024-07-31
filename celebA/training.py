from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from celebA.data.CelebA import CelebA
from celebA.models.archs.MobileNetV2 import build_model
from celebA.config import save_path, LABELS
from celebA.utils.loss_utils import calculating_class_weights, get_weighted_loss


def class_weights():
    # Build the dataset object
    celeba = CelebA(selected_features=LABELS)

    # Actual_labels
    y_true = celeba.attributes[LABELS].values

    # Get class weights
    class_weights = calculating_class_weights(y_true)

    return class_weights


# Training the model

if __name__ == "__main__":
    class_weights = class_weights()

    loss_name = 'weighted_loss'  # binary_crossentropy # binary_focal_crossentropy # TODO: Choose a loss function

    batch_size = 32
    num_epochs = 10
    weights_path = None  # TODO: path to pretrained weights

    # Hyper-params
    loss = get_weighted_loss(
        class_weights) if loss_name == 'weighted_loss' else loss_name  # keep name loss function for Keras arg

    # Augmentations for training set
    train_datagen = ImageDataGenerator(rotation_range=20,
                                       rescale=1. / 255,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    # Only rescaling the validation set
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # Get training and validation set:
    train_split = celeba.split('training', drop_zero=False)
    valid_split = celeba.split('validation', drop_zero=False)

    # Data generators:
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other'
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other'
    )

    # Build the model and compile
    model = build_model(num_features=celeba.num_features)
    model.summary()

    if weights_path is not None:
        model.load_weights(weights_path)

    model.compile(optimizer='adam', loss=get_weighted_loss(class_weights), metrics=['binary_accuracy'])

    # setup checkpoint callback:
    model_path = f"{save_path}/weights-FC{celeba.num_features}-MobileNetV2-{loss_name}.hdf5"

    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1)

    # Training:
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        max_queue_size=1,
        shuffle=True,
        callbacks=[checkpoint],
        verbose=1)

    # Get test set, and setup generator:
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = celeba.split('test', drop_zero=False)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_set,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='other')

    # Evaluate model:
    score = model.evaluate_generator(
        test_generator,
        steps=len(test_generator),
        max_queue_size=1,
        verbose=1)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model_path = f"{save_path}/weights-FC{celeba.num_features}-MobileNetV2" + "1epoch.hdf5"

# model = load_model("path/to/model.hd5f", custom_objects={"weighted_loss": get_weighted_loss(weights)}
