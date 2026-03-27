import numpy as np


def evaluate_model(model, x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)
    pred = model.predict(x)
    return float(np.mean(pred == y))

