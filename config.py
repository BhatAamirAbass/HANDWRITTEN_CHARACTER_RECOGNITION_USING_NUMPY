from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "emnist"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_emnist_model.keras"
RESULTS_DIR = PROJECT_DIR / "results"

NUM_CLASSES = 62
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHANNELS = 1
VALIDATION_SIZE = 0.10
RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001

TRAIN_IMAGES = "emnist-byclass-train-images-idx3-ubyte"
TRAIN_LABELS = "emnist-byclass-train-labels-idx1-ubyte"
TEST_IMAGES = "emnist-byclass-test-images-idx3-ubyte"
TEST_LABELS = "emnist-byclass-test-labels-idx1-ubyte"

CLASS_MAPPING = (
    [str(i) for i in range(10)]
    + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    + [chr(i) for i in range(ord("a"), ord("z") + 1)]
)
