import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE, EPOCHS, MODEL_DIR, MODEL_PATH,
    RANDOM_STATE, VALIDATION_SIZE
)
from dataset import load_dataset
from model import build_model
from preprocessing import preprocess_emnist_images


def main():
    (x_train_raw, y_train), (x_test_raw, y_test) = load_dataset()

    x_train = preprocess_emnist_images(x_train_raw)
    x_test = preprocess_emnist_images(x_test_raw)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    print("Training:", x_train.shape, y_train.shape)
    print("Validation:", x_val.shape, y_val.shape)
    print("Testing:", x_test.shape, y_test.shape)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    model = build_model()
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate the best checkpoint saved during training.
    best_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(
        x_test, y_test, batch_size=BATCH_SIZE, verbose=1
    )
    print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Final test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
