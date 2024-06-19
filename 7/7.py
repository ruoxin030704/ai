# 老師給的參考，問GPT解答，但還是不懂，所以以下全部複製
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 定義卷積層
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / in_channels)
        self.biases = np.zeros(out_channels)

    def forward(self, X):
        self.X = X
        batch_size, in_channels, in_height, in_width = X.shape
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (out_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # 在輸入圖像周圍進行填充
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(out_height):
            for j in range(out_width):
                X_slice = X_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                for k in range(self.out_channels):
                    out[:, k, i, j] = np.sum(X_slice * self.filters[k, :, :, :], axis=(1, 2, 3))

        out += self.biases[None, :, None, None]
        return out

    def backward(self, d_out, learning_rate):
        # 計算卷積層的梯度，並更新過濾器和偏差
        pass

# 定義池化層
class PoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        batch_size, channels, height, width = X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                X_slice = X[:, :, i * self.stride:i * self.stride + self.pool_size, j * self.stride:j * self.stride + self.pool_size]
                out[:, :, i, j] = np.max(X_slice, axis=(2, 3))

        return out

    def backward(self, d_out):
        # 計算池化層的梯度
        pass

# 定義全連接層
class FullyConnectedLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.biases = np.zeros(out_features)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.weights) + self.biases

    def backward(self, d_out, learning_rate):
        dX = np.dot(d_out, self.weights.T)
        dW = np.dot(self.X.T, d_out)
        db = np.sum(d_out, axis=0)

        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

        return dX

# 定義神經網路結構
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_out, learning_rate):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, learning_rate)

# 數據預處理函數
def preprocess_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].reshape(-1, 1, 28, 28)
    y = mnist['target'].astype(int)

    X = X / 255.0

    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    return train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 計算準確率函數
def compute_accuracy(predictions, labels):
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    return np.mean(pred_labels == true_labels)

# 訓練模型函數
def train():
    X_train, X_test, y_train, y_test = preprocess_data()

    model = NeuralNetwork()
    model.add(ConvLayer(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1))
    model.add(PoolingLayer(pool_size=2, stride=2))
    model.add(ConvLayer(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1))
    model.add(PoolingLayer(pool_size=2, stride=2))
    model.add(FullyConnectedLayer(in_features=16*7*7, out_features=128))
    model.add(FullyConnectedLayer(in_features=128, out_features=10))

    learning_rate = 0.01
    epochs = 10
    batch_size = 64

    for epoch in range(epochs):
        permutation = np.random.permutation(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # 前向傳播
            out = model.forward(X_batch)
            predictions = out.reshape(batch_size, -1)
            loss = -np.sum(y_batch * np.log(predictions)) / batch_size

            # 反向傳播
            d_out = (predictions - y_batch) / batch_size
            model.backward(d_out, learning_rate)

            total_loss += loss

        # 計算準確率
        train_acc = compute_accuracy(model.forward(X_train).reshape(len(X_train), -1), y_train)
        test_acc = compute_accuracy(model.forward(X_test).reshape(len(X_test), -1), y_test)

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(X_train)}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')

train()
