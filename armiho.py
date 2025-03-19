import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from aggregation import robust_aggregation
from attack import byzantine_attack

# 数据预处理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
x_train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
x_test_norm = np.linalg.norm(x_test, axis=1, keepdims=True)

x_train = x_train / x_train_norm
x_test = x_test / x_test_norm
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_noniid_data(x, y, num_clients, noniid_degree=0.5):
    num_classes = y.shape[1]
    data_per_client = len(x) // num_clients

    client_data = []
    class_indices = [np.where(np.argmax(y, axis=1) == i)[0] for i in range(num_classes)]

    for i in range(num_clients):
        client_indices = []
        for j in range(num_classes):
            amount = int(data_per_client * (1 - noniid_degree) / num_classes)
            client_indices.extend(class_indices[j][:amount])
            class_indices[j] = class_indices[j][amount:]

        client_indices.extend(class_indices[i][int(data_per_client * (1 - noniid_degree))-amount*(i+1):])
        np.random.shuffle(client_indices)
        client_data.append((x[client_indices], y[client_indices]))

    return client_data

# 定义逻辑回归模型
class LogisticRegression:
    def __init__(self, input_dim, output_dim):
        np.random.seed(2024)
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(1, output_dim)

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def compute_loss(self, X, y):
        m = X.shape[0]
        preds = self.predict(X)
        loss = -np.sum(y * np.log(preds + 1e-8)) / m
        return loss

    def compute_gradients(self, X, y):
        m = X.shape[0]
        preds = self.predict(X)
        grad_w = np.dot(X.T, (preds - y)) / m
        grad_b = np.sum(preds - y, axis=0, keepdims=True) / m
        return grad_w, grad_b

# 定义优化器
class Optimizer:
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v_w = 0
        self.v_b = 0
        self.alpha = 0.9

    def armijo_line_search(self, model, X, y, grad_w, grad_b, loss, c=0.5, tau=0.5):
        step_size = self.lr
        while True:
            new_weights = model.weights - step_size * grad_w
            new_bias = model.bias - step_size * grad_b
            model.weights, model.bias = new_weights, new_bias
            new_loss = model.compute_loss(X, y)
            if new_loss <= loss - c * step_size * (np.sum(grad_w**2) + np.sum(grad_b**2)):
                break
            step_size *= tau
        return step_size

    def update_weights(self, model, data_splits, start, end, last_grads, agg_type, regular_nodes, attack_type, grads_w, grads_b, method):
        num_nodes = len(data_splits)
        grads_w = [np.zeros_like(model.weights) for _ in range(num_nodes)]
        grads_b = [np.zeros_like(model.bias) for _ in range(num_nodes)]

        for node in range(num_nodes):
            X_node, y_node = data_splits[node]
            number_node = X_node.shape[0]
            X_batch = X_node[int(number_node * start):int(number_node * end)]
            y_batch = y_node[int(number_node * start):int(number_node * end)]
            if node in regular_nodes:
                grad_w, grad_b = model.compute_gradients(X_batch, y_batch)
            else:
                grad_w, grad_b = byzantine_attack(X_batch, y_batch, model.weights, model.bias, grads_w, grads_b, attack_type)
            if method == 'momentum' or method == 'nesterov_lookahead':
                grads_w[node] = self.momentum * grads_w[node] + (1 - self.momentum) * grad_w
                grads_b[node] = self.momentum * grads_b[node] + (1 - self.momentum) * grad_b
            else:
                grads_w[node] = grad_w
                grads_b[node] = grad_b

        grads = [np.concatenate((grads_w[i], grads_b[i]), axis=0) for i in range(num_nodes)]
        grad_agg = robust_aggregation(grads, agg_type, last_grads, regular_nodes)
        grad_w_agg = grad_agg[:784, :]
        grad_b_agg = grad_agg[-1, :]

        step_size = self.lr

        self.v_w = -grad_w_agg
        self.v_b = -grad_b_agg

        new_weights = model.weights + step_size * self.v_w
        new_bias = model.bias + step_size * self.v_b

        if method == 'nesterov_lookahead':
            new_weights = new_weights + self.momentum * (new_weights - model.weights)
            new_bias = new_bias + self.momentum * (new_bias - model.bias)

        return new_weights, new_bias, last_grads

# 分布式训练函数
def distributed_train_model(model, optimizer, X_train, y_train, X_test, y_test, epochs, num_nodes, batch_size, agg_type, attack_type, method):
    np.random.seed(2024)
    train_loss_history = []
    test_loss_history = []
    attacked_nodes = [8,9]
    regular_nodes = list(filter(lambda i: i not in attacked_nodes, range(num_nodes)))

    num_clients = 10
    noniid_degree = 0.5
    data_splits = create_noniid_data(X_train, y_train, num_clients, noniid_degree)

    X_train_regular = np.concatenate([data_splits[i][0] for i in range(8)], axis=0)
    y_train_regular = np.concatenate([data_splits[i][1] for i in range(8)], axis=0)
    train_loss = model.compute_loss(X_train, y_train)
    test_loss = model.compute_loss(X_test, y_test)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    last_grads = np.zeros_like(np.concatenate((model.weights, model.bias), axis=0))

    for epoch in range(epochs):
        num_batches = 1 // batch_size
        for i in range(int(num_batches)):
            start = i * batch_size
            end = start + batch_size
            model.weights, model.bias, last_grads = optimizer.update_weights(model, data_splits, start, end, last_grads, agg_type, regular_nodes, attack_type, None, None, method)

        train_loss = model.compute_loss(X_train_regular, y_train_regular)
        test_loss = model.compute_loss(X_test, y_test)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return train_loss_history, test_loss_history

# 初始化模型和优化器
input_dim = x_train.shape[1]
output_dim = 10
model = LogisticRegression(input_dim, output_dim)
optimizer = Optimizer(lr=1)

# 训练模型
epochs = 50
num_nodes = 10
momentum = 0.99
batch_size = 0.1

# for lr in range(100):
#     model_sgd = LogisticRegression(input_dim, output_dim)
#     optimizer_sgd = Optimizer(lr=2.21+0.01*lr) # 2.6(full) 0.5(120) 0.16(30) 0.1(0.05) 2.31(0.05)--0.3491
#     train_loss_sgd, test_loss_sgd = distributed_train_model(model_sgd, optimizer_sgd, x_train, y_train, x_test, y_test, epochs, num_nodes, batch_size, method = 'sgd')
#     print(2.21+0.01*lr, train_loss_sgd[-1])

# for lr in range(1000):
#     model_momentum = LogisticRegression(input_dim, output_dim)
#     optimizer_momentum = Optimizer(lr=8.15+0.01*lr) # 9.3 0.05 #8.29(0.05)--0.2722
#     train_loss_sgd, test_loss_sgd = distributed_train_model(model_momentum, optimizer_momentum, x_train, y_train, x_test, y_test, epochs, num_nodes, batch_size, method='momentum')
#     print(8.15+0.01*lr,train_loss_sgd[-1])

# for lr in range(1000):
#     model_nesterov_lookahead = LogisticRegression(input_dim, output_dim)
#     optimizer_nesterov_lookahead = Optimizer(lr=1, momentum=0.671295552041798) # 9.0 #8.63(0.05)--0.2714
#     train_loss_nesterov_lookahead, test_loss_nesterov_lookahead = distributed_train_model(model_nesterov_lookahead,
#                                                                               optimizer_nesterov_lookahead, x_train, y_train, x_test, y_test, epochs, num_nodes, batch_size, method = 'nesterov_lookahead')
#     print(1, train_loss_nesterov_lookahead[-1])

attack_types = {
    # "No Attack": 0,
    "Gaussian Attack": 1,
    "Sign Flipping": 2,
    "Label Flipping": 3,
    "Sample Duplicating": 4,
    "Zero Value": 5,
    "Isolation": 6,
    "A Little is Enough": 7,
    "IPM": 8,
    "AGRFang": 9,
    "Bit Flipping": 10
}

agg_types = {
    "Ideal": 14,
    # "Mean": 0,
    # "Median": 1,
    # "Trimmed Mean": 2,
    # "Geometric Median": 3,
    # "Krum": 4,
    # "Multi Krum": 5,
    # "FABA": 6,
    # "Remove Outliers": 7,
    # "Phocas": 8,
    # "Brute": 9,
    # "Bulyan": 10,
    # "Centered Clipping": 11,
    # "Sign Guard": 12,
    # "DNC": 13,
}
all_losses = {}

for label_att, attack_type in attack_types.items():
    i = 0
    for label_agg, agg_type in agg_types.items():
        i = i+1
        lr = 10
        model_sgd = LogisticRegression(input_dim, output_dim)
        optimizer_sgd = Optimizer(lr=lr) #2.6

        model_momentum = LogisticRegression(input_dim, output_dim)
        optimizer_momentum = Optimizer(lr=lr, momentum=momentum)

        print(f"Running scenario sgd: {label_agg}--{label_att}")
        train_loss_sgd, test_loss_sgd = distributed_train_model(model_sgd, optimizer_sgd, x_train, y_train, x_test, y_test, epochs, num_nodes, batch_size, agg_type, attack_type, method = 'sgd')
        with open('logr_outputsgd-det.txt', 'a') as file:
            file.write("(" + str(i)+','+ str(train_loss_sgd[-1]) + ')' +'\n')


        print(f"Running scenario momentum: {label_agg}--{label_att}")
        train_loss_momentum, test_loss_momentum = distributed_train_model(model_momentum, optimizer_momentum, x_train, y_train, x_test,
                                                                          y_test, epochs, num_nodes, batch_size, agg_type, attack_type, method='momentum')
        with open('logr_outputsgdm-det.txt', 'a') as file:
            file.write("(" + str(i)+','+ str(train_loss_momentum[-1]) + ')' +'\n')

        print(f"Running scenario nesterov: {label_agg}--{label_att}")
        # slope = (np.log(train_loss_sgd[20]) - np.log(train_loss_sgd[5]))/15
        # kappa = -2/slope
        # momentum = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)
        # print(momentum)
        model_nesterov_lookahead = LogisticRegression(input_dim, output_dim)
        optimizer_nesterov_lookahead = Optimizer(lr=lr, momentum=0.84) # 9.0
        train_loss_nesterov_lookahead, test_loss_nesterov_lookahead = distributed_train_model(model_nesterov_lookahead,
                                                                                              optimizer_nesterov_lookahead, x_train,
                                                                                              y_train, x_test, y_test, epochs, num_nodes, batch_size, agg_type, attack_type, method = 'nesterov_lookahead')
        with open('logr_outputsgdn-det.txt', 'a') as file:
            file.write("(" + str(i)+','+ str(train_loss_nesterov_lookahead[-1]) + ')' +'\n')

# 绘制损失曲线
# plt.plot(train_loss_sgd, label='Train Loss (Sgd)')
# plt.plot(train_loss_momentum, label='Train Loss (Momentum)')
# plt.plot(train_loss_nesterov_lookahead, label='Train Loss (Nesterov)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Loss Curves for Sgd vs. Momentum vs. Nesterov')
# plt.yscale('log')
# filename = 'Loss Curves for Sgd vs. Momentum vs. Nesterov - dis'+'.png'
# plt.savefig(filename)

