"""
knee_oa_classifier.py
---------------------
DenseNet201 fine-tuned for knee osteoarthritis severity grading.

Dataset: Knee Osteoarthritis Dataset with Severity
         https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

Classes (KL grading scale):
    0 — Normal
    1 — Doubtful
    2 — Minimal
    3 — Moderate
    4 — Severe

Usage:
    python knee_oa_classifier.py

Outputs:
    Best_DenseNet201.h5  — best checkpoint by val_accuracy
    accuracy/loss curves — displayed inline
    confusion matrix + classification report — printed to console
"""

import os
import random
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow
from tensorflow import keras
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pathlib


# =============================================================================
# PATHS
# =============================================================================

# keeping paths in one place — change base_dir here and everything else updates
BASE_DIR   = "/kaggle/input/knee-osteoarthritis-dataset-with-severity"
TRAIN_PATH = os.path.join(BASE_DIR, "train")
VALID_PATH = os.path.join(BASE_DIR, "val")
TEST_PATH  = os.path.join(BASE_DIR, "test")

os.chdir(BASE_DIR)


# =============================================================================
# VISUALISE A SAMPLE GRID
# =============================================================================

def show_sample_grid(train_path: str, n_classes: int = 5, n_cols: int = 5):
    # 5 rows x 5 cols — one row per KL grade, 5 random images per grade
    fig, ax = plt.subplots(n_classes, n_cols, figsize=(18, 18))
    for class_id in range(n_classes):
        folder  = os.path.join(train_path, str(class_id))
        samples = random.sample(os.listdir(folder), n_cols)
        for col in range(n_cols):
            image = cv2.imread(os.path.join(folder, samples[col]))
            ax[class_id, col].imshow(image)
            ax[class_id, col].set_title(f"class_{class_id}")
            ax[class_id, col].set_axis_off()
    plt.tight_layout()
    plt.show()


# =============================================================================
# CUSTOM AUGMENTATION: RANDOM ERASING
# =============================================================================

def random_erasing(img, sl=0.1, sh=0.2, rl=0.4, p=0.4):
    # randomly zeros out a rectangular patch to prevent over-reliance on local features
    # sl/sh control the erased area fraction, rl controls aspect ratio, p is apply probability
    h = tensorflow.shape(img)[0]
    w = tensorflow.shape(img)[1]
    c = tensorflow.shape(img)[2]
    origin_area = tensorflow.cast(h * w, tensorflow.float32)

    e_size_l = tensorflow.cast(tensorflow.round(tensorflow.sqrt(origin_area * sl * rl)), tensorflow.int32)
    e_size_h = tensorflow.cast(tensorflow.round(tensorflow.sqrt(origin_area * sh / rl)), tensorflow.int32)

    e_height_h = tensorflow.minimum(e_size_h, h)
    e_width_h  = tensorflow.minimum(e_size_h, w)

    erase_height = tensorflow.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tensorflow.int32)
    erase_width  = tensorflow.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h,  dtype=tensorflow.int32)

    erase_area = tensorflow.zeros(shape=[erase_height, erase_width, c])
    erase_area = tensorflow.cast(erase_area, tensorflow.uint8)

    pad_h      = h - erase_height
    pad_top    = tensorflow.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tensorflow.int32)
    pad_bottom = pad_h - pad_top

    pad_w     = w - erase_width
    pad_left  = tensorflow.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tensorflow.int32)
    pad_right = pad_w - pad_left

    # padding with 1s so multiplying by the mask zeroes out only the erased region
    erase_mask = tensorflow.pad(
        [erase_area],
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=1
    )
    erase_mask = tensorflow.squeeze(erase_mask, axis=0)
    erased_img = tensorflow.multiply(
        tensorflow.cast(img, tensorflow.float32),
        tensorflow.cast(erase_mask, tensorflow.float32)
    )

    # only apply erasing with probability p — otherwise return original image unchanged
    return tensorflow.cond(
        tensorflow.random.uniform([], 0, 1) > p,
        lambda: tensorflow.cast(img, img.dtype),
        lambda: tensorflow.cast(erased_img, img.dtype)
    )


# =============================================================================
# GPU SESSION
# =============================================================================

def init_gpu():
    # explicitly logging device placement — useful for debugging which ops land on GPU
    sess = tensorflow.compat.v1.Session(
        config=tensorflow.compat.v1.ConfigProto(log_device_placement=True)
    )
    return sess


# =============================================================================
# MODEL
# =============================================================================

def build_model() -> Model:
    # DenseNet201 pretrained on ImageNet — keeping all conv layers, replacing the classifier head
    base = tensorflow.keras.applications.DenseNet201(
        include_top=False,
        input_tensor=None,
        input_shape=None
    )

    x = base.output
    # GlobalAveragePooling instead of Flatten — fewer params, less overfitting on medical data
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.4)(x)

    # L1+L2 regularization on the final dense layer — this dataset isn't huge, regularize early
    predictions = Dense(
        units=5,
        activation="softmax",
        kernel_regularizer=regularizers.l1_l2(l1=0.02, l2=0.02)
    )(x)

    model = Model(inputs=base.input, outputs=predictions)

    # very low lr — fine-tuning a pretrained model, don't want to destroy the learned features
    model.compile(
        Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# =============================================================================
# DATA LOADERS
# =============================================================================

def get_dataloaders(train_path: str, valid_path: str, test_path: str, batch: int = 32):
    CLASSES = ["0", "1", "2", "3", "4"]

    # aggressive augmentation for train — X-ray images benefit from flips and rotations
    # random_erasing helps the model not fixate on specific bone texture regions
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.3,
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.15,
        zoom_range=0.2,
        preprocessing_function=random_erasing
    )

    # no augmentation at val/test time — just reading images as-is
    eval_gen = ImageDataGenerator()

    train_batches = train_gen.flow_from_directory(
        train_path, target_size=(224, 224), classes=CLASSES, batch_size=batch
    )
    valid_batches = eval_gen.flow_from_directory(
        valid_path, target_size=(224, 224), classes=CLASSES, batch_size=batch
    )
    # shuffle=False on test so predictions align with test_batches.classes for the confusion matrix
    test_batches = eval_gen.flow_from_directory(
        test_path, target_size=(224, 224), classes=CLASSES, batch_size=batch, shuffle=False
    )

    return train_batches, valid_batches, test_batches


# =============================================================================
# TRAINING
# =============================================================================

def train(model: Model, train_batches, valid_batches, batch: int = 32) -> dict:
    # saving best by val_accuracy — this is a classification task, accuracy is the right metric
    checkpoint = ModelCheckpoint(
        filepath="/kaggle/working/Best_DenseNet201.h5",
        verbose=2,
        save_best_only=True,
        monitor="val_accuracy"
    )

    # patience=20 because val_accuracy on 5-class medical data can plateau for many epochs
    early_stop = tensorflow.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=20,
        verbose=2,
        restore_best_weights=True
    )

    start = datetime.now()

    history = model.fit(
        train_batches,
        steps_per_epoch=5778 // batch,     # hardcoded dataset size — update if dataset changes
        validation_data=valid_batches,
        validation_steps=826 // batch,
        epochs=120,
        verbose=2,
        callbacks=[checkpoint, early_stop]
    )

    duration = datetime.now() - start
    print(f"Training completed in: {duration}")
    return history.history


# =============================================================================
# EVALUATION + PLOTS
# =============================================================================

def plot_history(history: dict):
    # accuracy curve
    plt.figure()
    plt.plot(history["accuracy"],     label="Train")
    plt.plot(history["val_accuracy"], label="Val")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # loss curve — checking these together to catch overfitting early
    plt.figure()
    plt.plot(history["loss"],     label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def evaluate_on_test(model: Model, test_batches, batch: int = 32):
    # test set eval — the number we actually report
    model.evaluate(test_batches)

    # predict_generator is deprecated in newer keras but keeping it for compatibility
    # +1 to the floor division so we don't miss the last partial batch
    Y_pred = model.predict_generator(test_batches, 1656 // batch + 1)
    y_pred = np.argmax(Y_pred, axis=1)

    print("\nConfusion Matrix")
    print(confusion_matrix(test_batches.classes, y_pred))

    print("\nClassification Report")
    target_names = ["0", "1", "2", "3", "4"]
    print(classification_report(test_batches.classes, y_pred, target_names=target_names))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    BATCH = 32

    # quick visual check before training — make sure images and labels look right
    show_sample_grid(TRAIN_PATH)

    init_gpu()

    model = build_model()

    train_batches, valid_batches, test_batches = get_dataloaders(
        TRAIN_PATH, VALID_PATH, TEST_PATH, batch=BATCH
    )

    history = train(model, train_batches, valid_batches, batch=BATCH)

    plot_history(history)

    evaluate_on_test(model, test_batches, batch=BATCH)
