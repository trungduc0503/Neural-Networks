import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load MNIST dataset
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape input
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Định nghĩa các hàm kích hoạt
def relu(x):
    return np.maximum(0, x)   # Hàm ReLU

def sigmoid(x):
    return 1. / (1. + np.exp(-x))    # Hàm sigmoid

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))     # Chuẩn hóa để tránh tràn số
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)      # Tính softmax

def forward_pass(X, Wh1, bh1, Wh2, bh2, Wo, bo):
    zh1 = X @ Wh1.T + bh1     # Tính đầu vào lớp ẩn 1
    a1 = relu(zh1)            # Kích hoạt bằng ReLU
    zh2 = a1 @ Wh2.T + bh2    # Tính đầu vào lớp ẩn 2
    a2 = sigmoid(zh2)         # Kích hoạt bằng sigmoid
    z = a2 @ Wo.T + bo        # Tính đầu vào lớp đầu ra
    o = softmax(z)            # Kích hoạt bằng softmax
    return a1, a2, o

def compute_accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# Hyperparameters
learning_rate = 0.5
epochs = 50
batch_size = 1000
num_batches = x_train.shape[0] // batch_size

# Network architecture
num_inputs = 784
num_hidden1 = 512
num_hidden2 = 256
num_classes = 10

# Khởi tạo trọng số và bias
Wh1 = np.random.uniform(-0.5, 0.5, (num_hidden1, num_inputs))
bh1 = np.zeros((1, num_hidden1))
Wh2 = np.random.uniform(-0.5, 0.5, (num_hidden2, num_hidden1))
bh2 = np.zeros((1, num_hidden2))
Wo = np.random.uniform(-0.5, 0.5, (num_classes, num_hidden2))
bo = np.zeros((1, num_classes))

losses = []
accuracies = []

# Training loop
for epoch in range(epochs):
    for i in range(num_batches):
        # Mini-batch selection
        x = x_train[i * batch_size:(i + 1) * batch_size]
        y = y_train[i * batch_size:(i + 1) * batch_size]
        
        # Lan truyền thuận
        a1, a2, o = forward_pass(x, Wh1, bh1, Wh2, bh2, Wo, bo)
        
        # Compute loss
        loss = -np.sum(y * np.log10(o + 1e-8))
        losses.append(loss)
        
        # Lan truyền ngược (backpropagation)
        d = o - y               # Gradient của hàm mất mát với softmax
        dWo = d.T @ a2
        dbo = np.mean(d)
        
        dh2 = (d @ Wo) * a2 * (1 - a2)     # Gradient qua lớp ẩn 2
        dWh2 = dh2.T @ a1
        dbh2 = np.mean(dh2)
        
        dh1 = (dh2 @ Wh2) * (a1 > 0)       # Gradient qua lớp ẩn 1
        dWh1 = dh1.T @ x
        dbh1 = np.mean(dh1)
        
        # Cập nhật trọng số
        Wo = Wo - learning_rate * dWo / batch_size
        bo = bo - learning_rate * dbo
        Wh2 = Wh2 - learning_rate * dWh2 / batch_size
        bh2 = bh2 - learning_rate * dbh2
        Wh1 = Wh1 - learning_rate * dWh1 / batch_size
        bh1 = bh1 - learning_rate * dbh1
    
    # Evaluate accuracy on test set
    _, _, test_pred = forward_pass(x_test, Wh1, bh1, Wh2, bh2, Wo, bo)
    accuracy = compute_accuracy(y_test, test_pred)
    accuracies.append(accuracy)
    
    # Visualize progress
    clear_output(wait=True)
    plt.plot(range(len(accuracies)), accuracies, 'o-',)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

print("Epochs:", epochs)
print("Final Accuracy:", accuracy)
