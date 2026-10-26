"""Train the LSTM on ACN-Data charging sessions and report real metrics.

Usage: python -m ml.train
Writes backend/ml/weights.npz and prints MAE/RMSE/MAPE on a held-out test split.
"""
import csv
import os

import numpy as np

from .lstm_numpy import LSTM, make_windows

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "public", "data", "acn-data.csv")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.npz")
WINDOW = 5
TEST_FRACTION = 0.2
EPOCHS = 300
LR = 0.01


def load_series(path=CSV_PATH, column="kWhDelivered"):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return np.array([float(row[column]) for row in reader])


def normalize(series):
    mean, std = series.mean(), series.std() or 1.0
    return (series - mean) / std, mean, std


def train(series=None, window=WINDOW, epochs=EPOCHS, lr=LR, seed=0):
    if series is None:
        series = load_series()
    norm, mean, std = normalize(series)
    X, y = make_windows(norm, window)

    n_test = max(1, int(len(X) * TEST_FRACTION))
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    model = LSTM(input_size=1, hidden_size=16, seed=seed)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(X_train))
        total_loss = 0.0
        for idx in order:
            total_loss += model.train_step(X_train[idx], y_train[idx], lr)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train_loss={total_loss / len(X_train):.5f}")

    preds_norm = np.array([model.predict(x) for x in X_test])
    preds = preds_norm * std + mean
    actual = y_test * std + mean

    mae = float(np.mean(np.abs(actual - preds)))
    rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - preds[nonzero]) / actual[nonzero])) * 100)

    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "n_test": int(n_test), "n_train": int(len(X_train))}
    model.save(WEIGHTS_PATH)
    model._norm_mean = mean
    model._norm_std = std
    return model, metrics


if __name__ == "__main__":
    _, metrics = train()
    print("\nTest-set metrics (held-out, not seen during training):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
