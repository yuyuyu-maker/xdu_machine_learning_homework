import numpy as np

from typing import Dict, Optional

from forward import linear, softmax


class SoftmaxRegression:
    """
    手写 softmax 回归
    """

    def __init__(self, in_features: int = 28 * 28, num_classes: int = 10, *, seed: Optional[int] = 42):
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)

        # 小随机初始化
        scale = 0.01
        rng = np.random.default_rng(seed if seed is not None else None)
        self.W = (scale * rng.standard_normal((self.in_features, self.num_classes))).astype(np.float32)
        self.b = np.zeros((self.num_classes,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: [N, D]
        return logits: [N, C]
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D [N,D], got shape={x.shape}")
        if x.shape[1] != self.in_features:
            raise ValueError(f"x has D={x.shape[1]} but expected {self.in_features}")
        return linear(x, self.W, self.b)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x), axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}
