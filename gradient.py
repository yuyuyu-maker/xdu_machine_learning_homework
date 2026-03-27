import numpy as np

from forward import one_hot, softmax


def softmax_regression_gradients(
    x: np.ndarray,
    logits: np.ndarray,
    y: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 softmax 回归的解析梯度。

    x: [N, D]
    logits: [N, C] = x@W + b
    y: [N]
    return: (dW: [D, C], db: [C])
    """
    x = np.asarray(x)
    logits = np.asarray(logits)
    y = np.asarray(y).astype(np.int64)
    n = x.shape[0]
    probs = softmax(logits, axis=1)  # [N, C]
    y_oh = one_hot(y, num_classes=num_classes).astype(probs.dtype, copy=False)  # [N, C]

    # dL/dlogits = (probs - one_hot) / N  (mean reduction)
    dlogits = (probs - y_oh) / n  # [N, C]
    dW = x.T @ dlogits  # [D, C]
    db = np.sum(dlogits, axis=0)  # [C]
    return dW, db
