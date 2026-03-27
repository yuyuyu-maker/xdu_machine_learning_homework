import gzip
import os
import struct
import urllib.error
import urllib.request

import numpy as np
from typing import Optional, Tuple
# 纯 NumPy 加载 MNIST：
# 1) 从官方地址下载 4 个 .gz 文件（训练图像/标签、测试图像/标签）
# 2) 按 IDX 二进制格式解析 header 和内容
# 3) 图像从 [N, 28, 28] 展平为 [N, 784]
# 4) 图像转 float32，标签转 int64；做 [0,1] 归一化

_BASE_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0 and _gzip_seems_ok(path):
        return
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

    last_err: Optional[Exception] = None
    for base in _BASE_URLS:
        url = base + filename
        try:
            _download_atomic(url, path, retries=2)
            if _gzip_seems_ok(path):
                return
            raise EOFError("downloaded gzip seems corrupted")
        except (urllib.error.HTTPError, urllib.error.URLError, EOFError, OSError) as e:
            last_err = e
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    raise RuntimeError(f"下载失败：{filename}。已尝试镜像：{_BASE_URLS}；最后错误：{last_err}")


def _download_atomic(url: str, path: str, *, retries: int = 2, chunk_size: int = 1 << 20):
    tmp = path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    last_err: Optional[Exception] = None
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    raise last_err if last_err is not None else RuntimeError("download failed")


def _gzip_seems_ok(path: str, *, chunk_size: int = 1 << 20) -> bool:
    """
    完整性校验：尝试把 gzip 流读到结尾。
    这样可以检测“文件开头正常，但尾部被截断”的情况（你现在遇到的 EOFError）。
    """
    try:
        with gzip.open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
        return True
    except Exception:
        return False


def _read_idx_images(gz_path: str) -> np.ndarray:
    """
    读取 IDX3 格式图像：
    - 前 16 字节：magic, 样本数, 行数, 列数（均为大端 uint32）
    - 后续字节：像素值（uint8）
    返回形状 [N, rows, cols] 的 uint8 数组。
    """
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad magic for images: {magic}")
        # 若 gzip 被截断，read 过程中会抛 EOFError
        data = f.read(n * rows * cols)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, rows, cols)
    return arr


def _read_idx_labels(gz_path: str) -> np.ndarray:
    """
    读取 IDX1 格式标签：
    - 前 8 字节：magic, 样本数（大端 uint32）
    - 后续字节：标签（uint8）
    返回形状 [N] 的 uint8 数组。
    """
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad magic for labels: {magic}")
        data = f.read(n)
    return np.frombuffer(data, dtype=np.uint8)


def load_mnist(root: str = "./data", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载并预处理 MNIST。

    处理逻辑：
    - 下载原始 gzip 文件到 root 目录
    - 解析出图像 [N,28,28] 与标签 [N]
    - 图像展平： [N,28,28] -> [N,784]
    - dtype 转换：图像 float32，标签 int64
    - 可选归一化：像素值从 [0,255] 缩放到 [0,1]

    返回：
      x_train: [60000, 784] float32
      y_train: [60000] int64
      x_test:  [10000, 784] float32
      y_test:  [10000] int64
    """
    paths = {k: os.path.join(root, v) for k, v in _FILES.items()}
    for k, filename in _FILES.items():
        _download(filename, paths[k])

    # 如果某个文件依然损坏（例如下载完成但被代理截断），这里读的时候会抛 EOFError；
    # 我们捕获后删除该文件并重下，再尝试一次。
    def _safe_read(reader, key: str):
        p = paths[key]
        try:
            return reader(p)
        except EOFError:
            try:
                os.remove(p)
            except OSError:
                pass
            _download(_FILES[key], p)
            return reader(p)

    x_train = _safe_read(_read_idx_images, "train_images")
    y_train = _safe_read(_read_idx_labels, "train_labels")
    x_test = _safe_read(_read_idx_images, "test_images")
    y_test = _safe_read(_read_idx_labels, "test_labels")

    # 展平图像：每张 28x28 压平成长度 784 的向量，便于线性分类器输入
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    return x_train, y_train, x_test, y_test


def train_val_split(x: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(val_ratio * n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

