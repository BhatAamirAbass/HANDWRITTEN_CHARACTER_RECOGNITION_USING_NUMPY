import struct
from pathlib import Path

import numpy as np

from config import (
    DATASET_DIR, TRAIN_IMAGES, TRAIN_LABELS,
    TEST_IMAGES, TEST_LABELS
)


def load_emnist_images(file_path):
    """Read an IDX3 image file and return uint8 images shaped (N, 28, 28)."""
    file_path = Path(file_path)
    with file_path.open("rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic number in {file_path}: {magic}")
        data = np.frombuffer(file.read(), dtype=np.uint8)

    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(
            f"Unexpected image data size in {file_path}: "
            f"expected {expected}, got {data.size}"
        )
    return data.reshape(count, rows, cols)


def load_emnist_labels(file_path):
    """Read an IDX1 label file and return uint8 labels shaped (N,)."""
    file_path = Path(file_path)
    with file_path.open("rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic number in {file_path}: {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if labels.size != count:
        raise ValueError(
            f"Unexpected label count in {file_path}: expected {count}, got {labels.size}"
        )
    return labels


def load_dataset(dataset_dir=DATASET_DIR):
    """Load the official EMNIST ByClass train and test files."""
    dataset_dir = Path(dataset_dir)
    required_files = [
        TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS
    ]
    missing = [name for name in required_files if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing EMNIST files in "
            f"{dataset_dir}:\n" + "\n".join(missing)
        )

    x_train = load_emnist_images(dataset_dir / TRAIN_IMAGES)
    y_train = load_emnist_labels(dataset_dir / TRAIN_LABELS)
    x_test = load_emnist_images(dataset_dir / TEST_IMAGES)
    y_test = load_emnist_labels(dataset_dir / TEST_LABELS)

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Image and label counts do not match.")

    return (x_train, y_train), (x_test, y_test)
