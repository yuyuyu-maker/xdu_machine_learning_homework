import numpy as np
#手写了一下这几个函数，在model中调用

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y.astype(np.int64)] = 1.0
    return out


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if keepdims:
        return y
    return np.squeeze(y, axis=axis)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits)
    m = np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits - m)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b