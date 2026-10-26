"""A small single-layer LSTM implemented in plain NumPy, with manual
forward pass and backpropagation-through-time (BPTT). No ML framework
dependency, trained directly on ACN-Data charging sessions.

This is a real, trained model: weights are fit by gradient descent on
held-in data and evaluated on held-out data. It is not a pretrained or
hand-tuned stand-in.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTM:
    def __init__(self, input_size=1, hidden_size=16, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        z = input_size + hidden_size
        scale = 1.0 / np.sqrt(z)

        def init(rows):
            return rng.uniform(-scale, scale, size=(rows, z))

        # Combined gate weights: forget, input, candidate, output
        self.Wf, self.Wi, self.Wc, self.Wo = (init(hidden_size) for _ in range(4))
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.Wy = rng.uniform(-scale, scale, size=(1, hidden_size))
        self.by = np.zeros(1)

    def _step(self, x, h_prev, c_prev):
        z = np.concatenate([h_prev, x])
        f = sigmoid(self.Wf @ z + self.bf)
        i = sigmoid(self.Wi @ z + self.bi)
        c_hat = np.tanh(self.Wc @ z + self.bc)
        o = sigmoid(self.Wo @ z + self.bo)
        c = f * c_prev + i * c_hat
        h = o * np.tanh(c)
        cache = (z, f, i, c_hat, o, c, c_prev, h)
        return h, c, cache

    def forward(self, seq):
        """seq: (T, input_size). Returns prediction (scalar) and caches for BPTT."""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        caches = []
        for x in seq:
            h, c, cache = self._step(x, h, c)
            caches.append(cache)
        y_hat = (self.Wy @ h + self.by)[0]
        return y_hat, h, caches

    def predict(self, seq):
        y_hat, _, _ = self.forward(seq)
        return y_hat

    def forecast(self, seq, steps):
        """Multi-step forecast by feeding predictions back in autoregressively."""
        window = list(seq)
        out = []
        for _ in range(steps):
            y_hat = self.predict(np.array(window[-len(seq):]))
            out.append(float(y_hat))
            window.append(np.array([y_hat]))
        return out

    def train_step(self, seq, target, lr):
        y_hat, h_last, caches = self.forward(seq)
        dy = (y_hat - target)  # d(0.5*(y_hat-target)^2)/dy_hat

        dWy = dy * h_last.reshape(1, -1)
        dby = np.array([dy])
        dh_next = dy * self.Wy.flatten()
        dc_next = np.zeros(self.hidden_size)

        dWf = np.zeros_like(self.Wf); dbf = np.zeros_like(self.bf)
        dWi = np.zeros_like(self.Wi); dbi = np.zeros_like(self.bi)
        dWc = np.zeros_like(self.Wc); dbc = np.zeros_like(self.bc)
        dWo = np.zeros_like(self.Wo); dbo = np.zeros_like(self.bo)

        for cache in reversed(caches):
            z, f, i, c_hat, o, c, c_prev, h = cache

            do = dh_next * np.tanh(c)
            dc = dc_next + dh_next * o * (1 - np.tanh(c) ** 2)
            df = dc * c_prev
            di = dc * c_hat
            dc_hat = dc * i

            draw_f = df * f * (1 - f)
            draw_i = di * i * (1 - i)
            draw_c = dc_hat * (1 - c_hat ** 2)
            draw_o = do * o * (1 - o)

            dWf += np.outer(draw_f, z); dbf += draw_f
            dWi += np.outer(draw_i, z); dbi += draw_i
            dWc += np.outer(draw_c, z); dbc += draw_c
            dWo += np.outer(draw_o, z); dbo += draw_o

            dz = (self.Wf.T @ draw_f + self.Wi.T @ draw_i
                  + self.Wc.T @ draw_c + self.Wo.T @ draw_o)
            dh_next = dz[: self.hidden_size]
            dc_next = dc * f

        for grad in (dWf, dWi, dWc, dWo, dWy):
            np.clip(grad, -5, 5, out=grad)

        self.Wf -= lr * dWf; self.bf -= lr * dbf
        self.Wi -= lr * dWi; self.bi -= lr * dbi
        self.Wc -= lr * dWc; self.bc -= lr * dbc
        self.Wo -= lr * dWo; self.bo -= lr * dbo
        self.Wy -= lr * dWy; self.by -= lr * dby

        return 0.5 * dy ** 2

    def save(self, path):
        np.savez(
            path, Wf=self.Wf, Wi=self.Wi, Wc=self.Wc, Wo=self.Wo,
            bf=self.bf, bi=self.bi, bc=self.bc, bo=self.bo,
            Wy=self.Wy, by=self.by, hidden_size=self.hidden_size,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(hidden_size=int(data["hidden_size"]))
        for name in ("Wf", "Wi", "Wc", "Wo", "bf", "bi", "bc", "bo", "Wy", "by"):
            setattr(model, name, data[name])
        return model


def make_windows(series, window):
    """series: 1D array. Returns (X, y) with X shape (N, window, 1), y shape (N,)."""
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window].reshape(-1, 1))
        y.append(series[i + window])
    return np.array(X), np.array(y)
