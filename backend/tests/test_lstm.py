import numpy as np

from ml.lstm_numpy import LSTM, make_windows


def test_make_windows_shapes():
    series = np.arange(10, dtype=float)
    X, y = make_windows(series, window=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    assert np.allclose(X[0].flatten(), [0, 1, 2])
    assert y[0] == 3


def test_training_reduces_loss_on_a_simple_pattern():
    # A trivially learnable pattern: predict the next value of a repeating ramp.
    series = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), 10)
    X, y = make_windows(series, window=3)
    model = LSTM(input_size=1, hidden_size=8, seed=1)

    losses = []
    for epoch in range(60):
        total = 0.0
        for i in range(len(X)):
            total += model.train_step(X[i], y[i], lr=0.05)
        losses.append(total / len(X))

    assert losses[-1] < losses[0]


def test_forecast_returns_requested_steps():
    model = LSTM(input_size=1, hidden_size=4, seed=2)
    seq = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    out = model.forecast(seq, steps=6)
    assert len(out) == 6
    assert all(isinstance(v, float) for v in out)


def test_save_and_load_roundtrip(tmp_path):
    model = LSTM(input_size=1, hidden_size=5, seed=3)
    seq = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    pred_before = model.predict(seq)

    path = tmp_path / "weights.npz"
    model.save(str(path))
    loaded = LSTM.load(str(path))
    pred_after = loaded.predict(seq)

    assert abs(pred_before - pred_after) < 1e-9
