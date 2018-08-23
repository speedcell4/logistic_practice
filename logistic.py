import numpy as np

from train import iteration, return_with_target


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Classifier(object):
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.W = np.random.random(input_size)
        self.b = np.array(0.)

    def forward(self, x):
        z = x @ self.W + self.b
        return sigmoid(z)

    def backward(self, x, y, t):  # backward = back propagation
        # 로지스틱 회귀의 최적화는 backward를 계산하는 것이며 이의 값은 광배법을 위한 값이 된다.
        # print(f'x.shape => {x.shape}')
        # print(f'y.shape => {y.shape}')
        # print(f't.shape => {t.shape}')
        return (y - t) @ x / x.shape[0], (y - t).mean()


def cross_entropy(y, t) -> float:
    # sometime y will be zero and then this results will come to be infinity, so we add a small enough
    # number (eps) to void this
    loss = -((t * np.log(y + np.finfo(np.float32).eps)) + (1. - t) * np.log(1. - y + np.finfo(np.float32).eps)).mean()
    return loss.__float__()
    # 곱하기 * 를 사용하면 sum을 사용해야 하지만 다 내적으로 곱해버리면 sum은 필요없음
    # x 내적 y = sum(x_i * y_i) 그렇지! 직접 행렬로 적어서 보면 아는 것을...


def accuracy_score(y, t) -> float:
    y = np.array([y_i > 0.5 for y_i in y], dtype=np.int32)
    t = np.array(t, dtype=np.int32)
    # manipulate data one np.ndarray level will always more easily to understand
    # your original code returns the mean of correct predicted probabilities, that does not make sense
    return (y == t).mean().__float__()


def train(vocab_size: int = 1000, epoch_num: int = 100, batch_size: int = 8, learning_rate: float = 0.1):
    data, targets = return_with_target(vocab_size)
    data = np.array(data, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    # print(f'data => {len(data)}\ntargets => {len(targets)}')
    # since we add another UNK token, the vocabulary size need to increase one
    logistic = Classifier(vocab_size + 1)

    for epoch in range(epoch_num):

        for x, t in iteration(data, targets, batch_size):
            y = logistic.forward(x)

            grad_W, grad_b = logistic.backward(x, y, t)
            logistic.W -= learning_rate * grad_W
            logistic.b -= learning_rate * grad_b

            # print(y)

        outputs = logistic.forward(data)
        acc_score = accuracy_score(outputs, targets)
        # print(acc_score)
        loss_score = cross_entropy(outputs, targets)
        print(f'[epoch {epoch:02d}] accuracy => {acc_score:.2f}, loss => {loss_score:.2f}')


if __name__ == '__main__':
    train()
