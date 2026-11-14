# Handwritten Character Recognition

A modular implementation of the EMNIST ByClass 62-class CNN project.

## Project structure

- `config.py`: paths, class mapping, and training hyperparameters
- `dataset.py`: IDX dataset readers and dataset loading
- `preprocessing.py`: exact EMNIST training/inference preprocessing
- `model.py`: original CNN architecture and compilation
- `train.py`: stratified validation split, class weights, callbacks, training
- `evaluate.py`: test evaluation and classification report
- `predict.py`: reusable prediction class for raw EMNIST-oriented images
- `app.py`: Flask drawing-canvas application
- `templates/`, `static/`: frontend files
- `data/raw/emnist/`: place the four official EMNIST ByClass IDX files here
- `models/`: place the trained Keras model here

## Important: preserve the trained model

Copy `best_emnist_model.keras` into `models/`. The web application loads it and does not retrain.

## Environment setup (Windows)

Use a 64-bit Python version supported by the TensorFlow release available for your platform. Python 3.11 is a practical starting point.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your installed Python/TensorFlow versions are already working, keep them rather than changing versions unnecessarily.

## Dataset placement

Put these four files inside `data/raw/emnist/`:

- `emnist-byclass-train-images-idx3-ubyte`
- `emnist-byclass-train-labels-idx1-ubyte`
- `emnist-byclass-test-images-idx3-ubyte`
- `emnist-byclass-test-labels-idx1-ubyte`

## Train from scratch (optional)

```bash
python train.py
```

This recreates the notebook's architecture and training setup. It is not required to run the application if the trained `.keras` model is already copied into `models/`.

## Evaluate the saved model

```bash
python evaluate.py
```

## Run the web application

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Faithfulness and caveats

The notebook's EMNIST preprocessing is reproduced exactly for the dataset: transpose axes `(0, 2, 1)`, convert to `float32`, divide by 255, and add the channel dimension. The training split uses `test_size=0.10`, `random_state=42`, and `stratify=y_train`. Class weights are computed from the training partition. The CNN architecture, Adam learning rate, loss, batch size, epochs, and callbacks follow the notebook.

The browser drawing canvas needs an adapter because a freehand browser drawing is not an original EMNIST IDX image. The current adapter crops, resizes, and centers a white-on-black drawing into 28×28. This is an initial heuristic, not proven equivalent to the EMNIST input distribution. Validate with real drawings and refine before claiming the benchmark test accuracy applies to user drawings.

The reported 82.51% full test accuracy and 82.90% random 1,000-image inference sample are historical results from the supplied project workbooks, not freshly reproduced by this refactor.
