# Animal Image Classification with CNN

A convolutional neural network for classifying images across 10 animal categories (Animals-10 dataset), built and trained in Google Colab with TensorFlow/Keras.

## What's actually in this repo

```text
cnn-10-animals/
├── README.md
├── LICENSE
├── src/
│   ├── data_preparation.ipynb   # downloads Animals-10 via Kaggle API, converts to
│   │                             # grayscale + RGB tensors (128×128), splits train/val/test
│   └── animals_cnn.ipynb        # builds and trains 4 models (see below), evaluates on
│                                 # the held-out test set
└── results/
    ├── training_curves.png      # accuracy/loss curves for the best-performing model
    ├── confusion_matrix.png     # confusion matrix on the test set
    └── sample_predictions.png   # example predictions, correct and incorrect
```

*(Two earlier placeholder files, `src/train.py` and `src/prepare_data.ipynb`, were removed — they were empty leftovers from an earlier repo layout and never contained code; all the actual training code lives in the two notebooks above.)*

## Models trained and actual results

`animals_cnn.ipynb` runs four experiments in sequence, in this order:

| # | Model | Input | Test Accuracy |
|---|---|---|---|
| 1 | MLP baseline (no convolution) | Grayscale, flattened | 20.67% (barely above the 10% random-guess baseline — included to show *why* a CNN is needed) |
| 2 | CNN | Grayscale | 65.28% |
| 3 | CNN | RGB | **68.60%** ← best verified result, shown in `results/` above |
| 4 | CNN (deeper, 4 conv blocks) | RGB | *Not verified* — this run's final test accuracy wasn't captured in the committed notebook output before it was saved. If you rerun the last cells and get a number here, update this table and swap in fresh result images. Until then, 68.60% is the only number this repo actually backs up. |

The three images in `results/` are the real, unedited outputs from experiment #3's evaluation cell — not regenerated or touched up.

## Architecture (best model, experiment #3)

```text
Input (128 × 128 × 3)
        ↓
Random Flip / Rotation / Zoom  (train-time only)
        ↓
Conv2D(32) + BatchNorm + MaxPool
        ↓
Conv2D(64) + BatchNorm + MaxPool
        ↓
Conv2D(128) + BatchNorm + MaxPool
        ↓
Global Average Pooling
        ↓
Dense(128, ReLU) + Dropout
        ↓
Dense(10, Softmax)
```

Trained with Adam (lr=0.001), sparse categorical cross-entropy, batch size 32, up to 30 epochs with early stopping (patience 8, monitoring validation loss).

## Running it

1. Open `src/data_preparation.ipynb` in Colab. You'll need a Kaggle API key (`kaggle.json`) to download the Animals-10 dataset — the notebook prompts for it. This produces `.npy` arrays saved to Google Drive.
2. Open `src/animals_cnn.ipynb` in Colab, pointed at the same Drive folder. Runs all four experiments and produces the evaluation plots.

## What I'd improve next

- Consolidate the three near-duplicate CNN-building cells in `animals_cnn.ipynb` into one parameterized function instead of copy-pasted blocks.
- Capture and commit the output of experiment #4 (or drop it if it doesn't add anything over #3).
- Try transfer learning from a pretrained backbone as a stronger baseline than training from scratch.
