"""
Lightweight NumPy inference for the EMNIST CNN.

TensorFlow is NOT needed at serving time. Render's free tier has 512 MB RAM and
a very small CPU share, which TensorFlow import + model load can exhaust.
The network is tiny, so plain NumPy runs it in a few milliseconds.
"""
import numpy as np

from config import CLASS_MAPPING, MODEL_DIR
from preprocessing import preprocess_single_emnist_image

WEIGHTS_PATH = MODEL_DIR / "weights.npz"


def _conv2d_relu(x, w, b):
    # x: (H, W, Cin), w: (kh, kw, Cin, Cout), 'valid' padding, stride 1
    kh, kw, cin, cout = w.shape
    h, wd, _ = x.shape
    oh, ow = h - kh + 1, wd - kw + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(0, 1))
    # windows: (oh, ow, Cin, kh, kw) -> (oh, ow, kh, kw, Cin)
    windows = windows.transpose(0, 1, 3, 4, 2).reshape(oh * ow, kh * kw * cin)
    out = windows @ w.reshape(kh * kw * cin, cout) + b
    return np.maximum(out, 0).reshape(oh, ow, cout)


def _maxpool2(x):
    h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    x = x[: h2 * 2, : w2 * 2]
    return x.reshape(h2, 2, w2, 2, c).max(axis=(1, 3))


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class CharacterPredictor:
    def __init__(self, weights_path=WEIGHTS_PATH):
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. Run `python export_weights.py` "
                "locally and commit models/weights.npz."
            )
        self.p = dict(np.load(weights_path))

    def _forward(self, image):
        p = self.p
        x = image.astype(np.float32)                      # (28, 28, 1)
        x = _maxpool2(_conv2d_relu(x, p["conv2d_w"], p["conv2d_b"]))
        x = _maxpool2(_conv2d_relu(x, p["conv2d_1_w"], p["conv2d_1_b"]))
        x = _conv2d_relu(x, p["conv2d_2_w"], p["conv2d_2_b"])
        x = x.reshape(-1)                                  # Flatten (H, W, C) order, same as Keras
        x = np.maximum(x @ p["dense_w"] + p["dense_b"], 0)
        return _softmax(x @ p["dense_1_w"] + p["dense_1_b"])

    def predict(self, raw_emnist_image):
        """Predict a raw 28x28 EMNIST-oriented grayscale image."""
        batch = preprocess_single_emnist_image(raw_emnist_image)  # (1, 28, 28, 1)
        scores = self._forward(batch[0])

        top_indices = np.argsort(scores)[-3:][::-1]
        best = int(top_indices[0])
        return {
            "character": CLASS_MAPPING[best],
            "class_index": best,
            "confidence": float(scores[best] * 100),
            "top_3": [
                {"character": CLASS_MAPPING[int(i)], "confidence": float(scores[i] * 100)}
                for i in top_indices
            ],
        }
