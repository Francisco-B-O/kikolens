"""Red neuronal con PyTorch para datos tabulares, como estimadores compatibles con scikit-learn. PyTorch es opcional
(`pip install kikolens[torch]`): este módulo se puede importar sin él y solo falla al entrenar. Así los mismos modelos
entran en la comparación del laboratorio, el ajuste de hiperparámetros y los paquetes de modelo."""
import importlib.util

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

NAME = "PyTorch MLP"


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _need_torch():
    if not torch_available():
        raise ImportError("PyTorch is not installed. Install it with: pip install kikolens[torch]")


class _TorchTabular(BaseEstimator):
    """Perceptrón multicapa con Adam, abandono (dropout), decaimiento de pesos y parada temprana sobre un 15 % reservado."""

    def __init__(self, hidden=(64, 32), dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=64, patience=15,
                 seed=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def _network(self, n_in: int, n_out: int):
        import torch.nn as nn
        layers, width = [], n_in
        for size in self.hidden:
            layers += [nn.Linear(width, int(size)), nn.ReLU(), nn.Dropout(self.dropout)]
            width = int(size)
        layers.append(nn.Linear(width, n_out))
        return nn.Sequential(*layers)

    def _fit_network(self, X, target, loss_fn, n_out: int) -> None:
        _need_torch()
        import torch
        if self.epochs < 1 or self.batch_size < 1 or not 0 <= self.dropout < 1:
            raise ValueError("epochs and batch_size must be at least 1 and dropout must be in [0, 1).")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        n = len(X)
        order = rng.permutation(n)
        n_val = int(0.15 * n) if n >= 40 else 0
        val, train = order[:n_val], order[n_val:]
        net = self._network(X.shape[1], n_out)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, wait, history = np.inf, None, 0, []
        for _ in range(self.epochs):
            net.train()
            shuffled = rng.permutation(train)
            for start in range(0, len(shuffled), self.batch_size):
                batch = torch.from_numpy(shuffled[start:start + self.batch_size])
                optimizer.zero_grad()
                loss_fn(net(X[batch]), target[batch]).backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                check = val if n_val else train
                score = float(loss_fn(net(X[check]), target[check]))
            history.append(score)
            if score < best - 1e-6:
                best, best_state, wait = score, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_, self.history_, self.n_features_in_ = net, history, X.shape[1]

    def _forward(self, X):
        import torch
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} columns, got {X.shape[1]}.")
        with torch.no_grad():
            return self.net_(torch.from_numpy(X)).numpy()


class TorchMLPRegressor(RegressorMixin, _TorchTabular):
    def fit(self, X, y):
        _need_torch()
        import torch
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if not np.isfinite(y).all():
            raise ValueError("The target has missing or infinite values.")
        self._fit_network(X, torch.from_numpy(y), torch.nn.MSELoss(), 1)
        return self

    def predict(self, X):
        return self._forward(X).ravel()


class TorchMLPClassifier(ClassifierMixin, _TorchTabular):
    def fit(self, X, y):
        _need_torch()
        import torch
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Classification needs at least 2 classes.")
        self._fit_network(X, torch.from_numpy(codes.astype(np.int64)), torch.nn.CrossEntropyLoss(), len(self.classes_))
        return self

    def predict_proba(self, X):
        logits = self._forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
