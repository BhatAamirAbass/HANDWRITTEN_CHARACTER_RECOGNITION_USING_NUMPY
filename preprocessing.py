import numpy as np


def preprocess_emnist_images(images):
    """
    Apply the exact training preprocessing from the exploration notebook:
    transpose axes 1 and 2, cast to float32, divide by 255, add channel axis.
    """
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Expected (N, 28, 28) images, received {images.shape}")

    images = np.transpose(images, (0, 2, 1))
    images = images.astype(np.float32) / 255.0
    return np.expand_dims(images, axis=-1)


def preprocess_single_emnist_image(image):
    """Preprocess one raw uint8 EMNIST image using the training pipeline."""
    image = np.asarray(image)
    if image.shape != (28, 28):
        raise ValueError(f"Expected (28, 28) image, received {image.shape}")
    return preprocess_emnist_images(image[np.newaxis, ...])
