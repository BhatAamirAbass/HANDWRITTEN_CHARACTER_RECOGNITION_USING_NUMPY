import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, NUM_CLASSES, LEARNING_RATE


def build_model():
    """Recreate the CNN architecture recorded in 01_Dataset_Exploration."""
    model = models.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
