import numpy as np
from scipy.stats import norm, trim_mean

def predict(X, weights, bias):
    logits = np.dot(X, weights) + bias
    return softmax(logits)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def compute_loss(self, X, y, weights, bias):
    m = X.shape[0]
    preds = self.predict(X, weights, bias)
    loss = -np.sum(y * np.log(preds + 1e-8)) / m
    return loss

def compute_gradients(weights, bias, X, y):
    m = X.shape[0]
    preds = predict(X, weights, bias)
    grad_w = np.dot(X.T, (preds - y)) / m
    grad_b = np.sum(preds - y, axis=0, keepdims=True) / m
    return grad_w, grad_b

def byzantine_attack(local_x, local_y, w_lookahead, b_lookahead, grads_w, grads_b, attack_type):
    num_regular = 8
    num_workers = 10
    # 没有攻击
    if attack_type == 0:
        grad_w, grad_b = compute_gradients(local_x, local_y, w_lookahead, b_lookahead)

    # 仅在特定节点上添加高斯噪声
    if attack_type == 1:
        grad_w, grad_b = compute_gradients(w_lookahead, b_lookahead, local_x, local_y)
        grad_w = add_gaussian_noise(grad_w, stddev=0.5)
        grad_b = add_gaussian_noise(grad_b, stddev=0.5)

    # 应用Sign Flipping攻击
    if attack_type == 2:
        mean_grad_w = np.mean(grads_w[:8], axis=0)
        mean_grad_b = np.mean(grads_b[:8], axis=0)
        grad_w = sign_flipping_attack(mean_grad_w)
        grad_b = sign_flipping_attack(mean_grad_b)

    # 应用Label Flipping攻击
    if attack_type == 3:
        grad_w, grad_b = compute_gradients(w_lookahead, b_lookahead, local_x, 1-local_y)

    # 应用Sample Duplicating攻击
    if attack_type == 4:
        grad_w, grad_b = grads_w[0], grads_b[0]

    # 应用Zero Value攻击
    if attack_type == 5:
        grad_w, grad_b = np.zeros_like(w_lookahead), np.zeros_like(b_lookahead)

    # 应用Isolation攻击
    if attack_type == 6:
        num_attack = 2
        indices = np.random.choice(num_regular, num_attack, replace=False)
        grad_w, grad_b = isolation_attack(grads_w[:8],indices), isolation_attack(grads_b[:8],indices)

    # 应用A Little is Enough攻击
    if attack_type == 7:
        grad_w = alittleisenough_attack(grads_w[:8], num_regular, num_workers, z = 1)
        grad_b = alittleisenough_attack(grads_b[:8], num_regular, num_workers, z = 1)

    # 应用IPM攻击
    if attack_type == 8:
        epsilon = 0.01
        num_attack = 2
        indices = np.random.choice(num_regular, num_attack, replace=False)
        grad_w, grad_b = isolation_attack(grads_w[:8],indices) + epsilon, isolation_attack(grads_b[:8],indices) + epsilon

    # 应用AGRFang攻击
    if attack_type == 9:
        agr_scale = 0.05
        grad_w, grad_b = AGRFang_attack(grads_w[:8], num_regular, agr_scale), AGRFang_attack(grads_b[:8], num_regular, agr_scale)

    # 应用Bit Flipping攻击
    if attack_type == 10:
        grad_w, grad_b = compute_gradients(w_lookahead, b_lookahead, local_x, local_y)
        grad_w, grad_b = - grad_w, - grad_b

    return grad_w, grad_b


# 添加高斯噪声
def add_gaussian_noise(grad, mean=0.0, stddev=0.1):
    noise = np.random.normal(mean, stddev, grad.shape)
    return grad + noise

# Sign Flipping攻击
def sign_flipping_attack(grad, sign_scale=-4):
    return sign_scale * grad

# Isolation攻击
def isolation_attack(grads, indices):
    grad_attack = [grads[i] for i in indices]
    return - np.sum(grad_attack, axis=0)

# A Little is Enough攻击
def alittleisenough_attack(grads, num_regular, num_workers, z = None):
    if z is not None:
        z_max = z
    else:
        s = np.floor(num_workers / 2 + 1) - num_workers + num_regular
        cdf_value = (num_regular - s) / num_regular
        z_max = norm.ppf(cdf_value)

    mu = np.mean(grads[:num_regular], axis=0)
    std = np.std(grads[:num_regular])

    return mu - std * z_max

# AGRFang攻击
def AGRFang_attack(grads, num_regular, agr_scale = 0.1):
    mu = np.mean(grads[:num_regular], axis=0)
    mu_sign = np.sign(mu)

    return mu - mu_sign * agr_scale