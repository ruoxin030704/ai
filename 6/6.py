# 老師給的參考，問GPT解答，但還是不懂，所以以下全部複製
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward

        return out

    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=1)[:, None])
        out = Tensor(exps / np.sum(exps, axis=1)[:, None], (self,), 'softmax')

        def _backward():
            s = np.sum(out.grad * out.data, axis=1)[:, None]
            self.grad += out.data * (out.grad - s)
        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward

        return out

    def sum(self, axis=None):
        out = Tensor(np.sum(self.data, axis=axis), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out

    def cross_entropy(self, target):
        log_probs = self.log()
        loss = -np.sum(target.data * log_probs.data, axis=1)
        out = Tensor(np.mean(loss), (self, target), 'cross_entropy')

        def _backward():
            self.grad += (self.data - target.data) / self.data.shape[0]
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self.bias = Tensor(np.zeros(out_features))

    def __call__(self, x):
        return x.matmul(self.weights) + self.bias

def preprocess_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target'].astype(int)

    X = X / 255.0

    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    return train_test_split(X, y_onehot, test_size=0.2, random_state=42)

def train():
    X_train, X_test, y_train, y_test = preprocess_data()

    fc1 = Linear(784, 128)
    fc2 = Linear(128, 10)

    learning_rate = 0.01
    for epoch in range(10):
        total_loss = 0
        for x, y in zip(X_train, y_train):
            x = Tensor(x)
            y = Tensor(y)
            h1 = fc1(x).relu()
            logits = fc2(h1)
            y_pred = logits.softmax()

            loss = y_pred.cross_entropy(y)

            loss.backward()

            for layer in [fc1, fc2]:
                layer.weights.data -= learning_rate * layer.weights.grad
                layer.bias.data -= learning_rate * layer.bias.grad
                layer.weights.grad = np.zeros_like(layer.weights.grad)
                layer.bias.grad = np.zeros_like(layer.bias.grad)

            total_loss += loss.data

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(X_train)}')

train()
