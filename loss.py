import numpy as np

from forward import logsumexp


def cross_entropy_loss(logits: np.ndarray, y: np.ndarray, reduction: str = "mean") -> float | np.ndarray:
    """
    手写 softmax + 交叉熵（不使用 torch.nn.functional / autograd）。

    logits: [N, C]
    y: [N] (long)
    reduction: "mean" | "sum" | "none"
    """
    logits = np.asarray(logits)
    y = np.asarray(y).astype(np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [N,C], got shape={logits.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D [N], got shape={y.shape}")

    # 交叉熵损失函数-log p(y|x) = -log softmax(logits)[range(N), y]
    lse = logsumexp(logits, axis=1, keepdims=False)  # [N]
    correct = logits[np.arange(logits.shape[0]), y]  # [N]
    loss = lse - correct  # [N]

    if reduction == "none":
        return loss
    if reduction == "sum":
        return float(np.sum(loss))
    if reduction == "mean":
        return float(np.mean(loss))
    raise ValueError(f"Unknown reduction={reduction!r}")
