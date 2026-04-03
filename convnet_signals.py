# convnet_signals.py
# ConvNet on SignalsDatasets using Keras (Python/TensorFlow)
# Install dependencies: pip install tensorflow pandas numpy

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Load dataset ──────────────────────────────────────────────────────────────
train_signal = pd.read_csv("train_signal.csv")

# ── Image configuration ───────────────────────────────────────────────────────
# Adjust these to match your actual preprocessed data dimensions
img_width  = 20
img_height = 20
channels   = 3

# Input shape used in the original model (64 x 64 x 5)
INPUT_SHAPE = (64, 64, 5)

# ── Build the ConvNet ─────────────────────────────────────────────────────────
model = keras.Sequential([
    # Block 1 — Conv → Leaky ReLU → BatchNorm → MaxPool
    layers.Conv2D(filters=128, kernel_size=(5, 5), activation="relu",
                  input_shape=INPUT_SHAPE),
    layers.LeakyReLU(negative_slope=0.5),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Block 2 — Conv → MaxPool
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Block 3 — Conv → Flatten → ReLU
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
    layers.Flatten(),
    layers.Activation("relu"),
])

model.summary()

# ── Compile ───────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Training (uncomment and adjust arrays to your data) ──────────────────────
# train_x should be a numpy array shaped (n_samples, 64, 64, 5)
# train_y should be a numpy array shaped (n_samples,) with integer class labels
#
# history = model.fit(
#     x=train_x,
#     y=train_y,
#     epochs=10,
#     validation_data=(val_x, val_y),
#     verbose=2,
# )
