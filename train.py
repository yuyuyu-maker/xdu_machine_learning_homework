import numpy as np
from data import load_mnist, train_val_split#数据加载
from evaluate import evaluate_model, confusion_matrix, plot_confusion_matrix, plot_training_curves#评估
from gradient import softmax_regression_gradients#梯度  
from loss import cross_entropy_loss#损失    
from model import SoftmaxRegression#模型
#说明一下，其实这个模块化做的不是很好，只不过我现在想到的是这样的，所以先这样写了


def iter_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool = True, seed: int = 42):
    """
    伪代码：
    1. 生成样本索引 idx = [0..N-1]
    2. 若 shuffle=True，则按 seed 打乱 idx
    3. 按 batch_size 切片 idx，产出 x[idx_batch], y[idx_batch]
    """
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield x[b], y[b]


def train_model(
    model: SoftmaxRegression,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float = 0.1,
    epochs: int = 5,
    batch_size: int = 100,
):
    """
    伪代码（训练主循环）：
    1. for epoch in 1..epochs:
       a) 清空本轮统计量（loss/准确率计数）
       b) 遍历 mini-batch：
          - 前向：logits = model.forward(xb)
          - 损失：loss = cross_entropy_loss(logits, yb)
          - 反向：dW, db = softmax_regression_gradients(...)
          - 更新：W -= lr*dW, b -= lr*db
          - 累计训练集 loss 与准确率
       c) 在验证集上评估准确率
       d) 打印 epoch 指标
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    for epoch in range(1, epochs + 1):
        if epoch%10==0:
            print(f"Epoch {epoch} of {epochs} is running")
            #只是用来看看有没有开始运行
        running_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in iter_minibatches(x_train, y_train, batch_size, shuffle=True, seed=epoch):
            logits = model.forward(xb)
            loss = cross_entropy_loss(logits, yb, reduction="mean")
            dW, db = softmax_regression_gradients(xb, logits, yb, num_classes=model.num_classes)

            model.W = model.W - lr * dW
            model.b = model.b - lr * db

            running_loss += float(loss) * xb.shape[0]
            total += xb.shape[0]
            pred = np.argmax(logits, axis=1)
            correct += int(np.sum(pred == yb))

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = evaluate_model(model, x_val, y_val)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    return history


if __name__ == "__main__":
    print("开始加载数据")
    x_train, y_train, x_test, y_test = load_mnist(root="./data", normalize=True)
    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_ratio=0.2, seed=42)
    print("数据加载完成")
    model = SoftmaxRegression(in_features=28 * 28, num_classes=10, seed=42)
    history = train_model(model, x_train, y_train, x_val, y_val, lr=0.1, epochs=5, batch_size=100)
    test_acc = evaluate_model(model, x_test, y_test)
    print("final_test_acc=", test_acc)

    # 记录训练曲线：loss 与 accuracy
    plot_training_curves(history, save_path="training_curves.png")

    # 绘制混淆矩阵
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    plot_confusion_matrix(cm, save_path="confusion_matrix.png")
    print("图像已保存: training_curves.png, confusion_matrix.png")