import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_MAPPING, MODEL_PATH, BATCH_SIZE
from dataset import load_dataset
from preprocessing import preprocess_emnist_images


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    (_, _), (x_test_raw, y_test) = load_dataset()
    x_test = preprocess_emnist_images(x_test_raw)
    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print(f"Test loss: {loss:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=np.arange(62),
        target_names=CLASS_MAPPING,
        zero_division=0,
    ))

    matrix = confusion_matrix(y_test, predictions, labels=np.arange(62))
    print("\nConfusion matrix shape:", matrix.shape)


if __name__ == "__main__":
    main()
