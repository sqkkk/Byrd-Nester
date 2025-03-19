import numpy as np
from scipy.stats import norm, trim_mean
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools
from sklearn.cluster import MeanShift

# 聚合器
def robust_aggregation(grads, agg_type, last_grad,regular_nodes):

    # Mean
    if agg_type == 0:
        return np.mean(grads, axis =0)

    # Median
    if agg_type == 1:
        return np.median(grads, axis=0)

    # Trimmed Mean
    if agg_type == 2:
        proportion_to_cut = 0.3
        return trim_mean(grads, proportion_to_cut, axis=0)

    # Geometric Median
    if agg_type == 3:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return gm(grads_array).reshape((-1, m))

    # Krum
    if agg_type == 4:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return krum(grads_array,1).reshape((-1, m))

    # Multi Krum
    if agg_type == 5:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return multi_krum(grads_array,1,8).reshape((-1, m))

    # FABA
    if agg_type == 6:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return faba(grads_array,4).reshape((-1, m))

    # Remove Outliers
    if agg_type == 7:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return remove_outliers(grads_array,4).reshape((-1, m))

    # Phocas
    if agg_type == 8:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return phocas(grads_array,2).reshape((-1, m))

    # Brute
    if agg_type == 9:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return brute(grads_array,2).reshape((-1, m))

    # Bulyan
    if agg_type == 10:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return bulyan(grads_array,2).reshape((-1, m))

    # centered_clipping
    if agg_type == 11:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return centered_clipping(grads_array, last_grad, n_iter = 1).reshape((-1, m))

    # sign guard
    if agg_type == 12:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return sign_guard(grads_array).reshape((-1, m))

    # dnc agr
    if agg_type == 13:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return dnc_agr(grads_array).reshape((-1, m))

    # god
    if agg_type == 14:
        m = len(grads)
        grads_array = np.array(grads).reshape((m, -1))
        return np.mean(grads_array[regular_nodes], axis=0).reshape((-1, m))



def gm(wList):
    max_iter = 1000
    tol = 1e-5
    guess = np.mean(wList, axis=0)
    for _ in range(max_iter):
        dist_li = np.linalg.norm(wList - guess, axis=1)
        dist_li[dist_li == 0] = 1
        temp1 = np.sum(np.stack([w/d for w, d in zip(wList, dist_li)]), axis=0)
        temp2 = np.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = np.linalg.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
    return guess


def geometric_median(X, eps=1e-5):
    # 初始猜测值，使用均值作为起始点
    y = np.mean(X, axis=0)

    # 定义目标函数：到所有点的距离和
    def objective_function(y_flat):
        y = y_flat.reshape(X.shape[1:])
        return np.sum(np.linalg.norm(X - y, axis=1))

    # 使用 scipy.optimize.minimize 来优化几何中位数
    y_flat = y.flatten()
    options = {'maxiter': 100}
    result = minimize(objective_function, y_flat, method='BFGS', tol=10, options=options)
    return result.x.reshape(X.shape[1:])

# Krum
def krum(grads, f):
    """
    使用 Krum 聚合规则选择代表性梯度

    参数:
    gradients: numpy 数组，形状为 (n, d)，其中 n 是节点数，d 是梯度的维度
    f: 可容忍的拜占庭节点数

    返回:
    选择的代表性梯度，形状为 (d,)
    """
    n = grads.shape[0]

    # 检查输入是否有效
    if n <= 2 * f:
        raise ValueError("节点数量必须大于 2f 才能保证 Krum 的正确性")

    # 计算每个梯度与其他梯度的欧氏距离
    distances = cdist(grads, grads, 'euclidean')

    # 对每个节点计算其与其他节点的最小距离和
    scores = []
    for i in range(n):
        # 排除自身的距离
        sorted_distances = np.sort(distances[i])
        score = np.sum(sorted_distances[1:n-f])
        scores.append(score)

    # 选择得分最小的梯度
    selected_index = np.argmin(scores)
    return grads[selected_index]

# Multi Krum
def multi_krum(gradients, f, m):
    """
    使用 Multi-Krum 聚合规则选择代表性梯度

    参数:
    gradients: numpy 数组，形状为 (n, d)，其中 n 是节点数，d 是梯度的维度
    f: 可容忍的拜占庭节点数
    m: 选择的代表性梯度数

    返回:
    聚合后的梯度，形状为 (d,)
    """
    n = gradients.shape[0]

    # 检查输入是否有效
    if n <= 2 * f:
        raise ValueError("节点数量必须大于 2f 才能保证 Multi-Krum 的正确性")
    if m > n - 2 * f:
        raise ValueError("选取的代表性梯度数 m 必须小于等于 n - 2f")

    # 计算每个梯度与其他梯度的欧氏距离
    distances = cdist(gradients, gradients, 'euclidean')

    # 对每个节点计算其与其他节点的最小距离和
    scores = []
    for i in range(n):
        sorted_distances = np.sort(distances[i])
        score = np.sum(sorted_distances[1:n-f-1])
        scores.append((score, i))

    # 选择得分最小的前 m 个梯度
    scores.sort()
    selected_indices = [scores[i][1] for i in range(m)]
    selected_gradients = gradients[selected_indices]

    # 对选定的梯度进行平均
    aggregated_gradient = np.mean(selected_gradients, axis=0)

    return aggregated_gradient

# FABA
def faba(grads, f):
    """
    使用 FABA 聚合规则筛选并聚合梯度

    参数:
    gradients: numpy 数组，形状为 (n, d)，其中 n 是节点数，d 是梯度的维度
    f: 可容忍的拜占庭节点数

    返回:
    聚合后的梯度，形状为 (d,)
    """
    n = grads.shape[0]

    # 检查输入是否有效
    if n <= 2 * f:
        raise ValueError("节点数量必须大于 2f 才能保证 FABA 的正确性")

    # 初始候选集
    candidates = np.copy(grads)

    # 迭代去除异常梯度
    for _ in range(f):
        # 计算候选梯度之间的欧氏距离
        distances = cdist(candidates, candidates, 'euclidean')

        # 计算每个梯度与其他梯度的距离和
        scores = np.sum(distances, axis=1)

        # 识别距离和最大的梯度索引
        to_remove = np.argmax(scores)

        # 去除该梯度
        candidates = np.delete(candidates, to_remove, axis=0)

    # 对剩余的候选梯度进行平均
    aggregated_gradient = np.mean(candidates, axis=0)

    return aggregated_gradient

# Remove Outliers
def remove_outliers(gradients, f):
    """
    使用 Remove Outliers 聚合规则筛选并聚合梯度

    参数:
    gradients: numpy 数组，形状为 (n, d)，其中 n 是节点数，d 是梯度的维度
    f: 可容忍的拜占庭节点数，即要去除的最远数据的数量

    返回:
    聚合后的梯度，形状为 (d,)
    """
    n = gradients.shape[0]

    # 检查输入是否有效
    if f >= n:
        raise ValueError("f 必须小于节点数 n")

    # 计算所有梯度的平均值
    mean_gradient = np.mean(gradients, axis=0)

    # 计算每个梯度与平均值的欧氏距离
    distances = np.linalg.norm(gradients - mean_gradient, axis=1)

    # 按距离排序并去除距离最大的 f 个梯度
    farthest_indices = np.argsort(distances)[-f:]
    valid_gradients = np.delete(gradients, farthest_indices, axis=0)

    # 对剩余的有效梯度进行平均
    aggregated_gradient = np.mean(valid_gradients, axis=0)

    return aggregated_gradient

# Phocas
def phocas(grads, f):
    """
    使用 Phocas 聚合规则去掉 f 个离群点并聚合梯度

    参数:
    neighbor_messages: torch.Tensor，形状为 (n, d)，其中 n 是节点数，d 是梯度的维度
    byzantine_size: 可容忍的拜占庭节点数，即要去除的最远数据的数量

    返回:
    聚合后的梯度，形状为 (d,)
    """
    remain = np.copy(grads)

    if remain.shape[0] <= 1:
        return np.mean(remain, axis=0)

    if 2 * f >= len(grads):
        f = max(len(grads) // 2 - 1, 0)

    # 计算去掉 f 个极端值后的均值
    proportion_to_cut = f/len(grads)
    mean = trim_mean(remain, proportion_to_cut, axis=0)

    # 计算每个梯度与均值的距离
    distances = np.linalg.norm(remain - mean, axis=1)

    # 移除距离最大的 f 个梯度
    for _ in range(f):
        if remain.shape[0] <= 1:
            break
        remove_index = np.argmax(distances)
        remain = np.delete(remain, remove_index, axis=0)
        distances = np.delete(distances, remove_index)

    return np.mean(remain, axis=0)

# Brute
def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
      data Indexable (including ability to query length) containing the elements
    Returns:
      Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield data[i], data[j]

def brute_selection(grads, f):
    """ Brute rule.
    Brute is also called minimum diameter averaging (MDA).
    The code is simplified for clarity.

    Args:
      neighbor_messages Non-empty list of neighbor_messages to aggregate
      f         Number of Byzantine neighbor_messages to tolerate
    Returns:
      Selection index set
    """
    n = len(grads)
    # Compute all pairwise distances
    distances = np.zeros(n * (n - 1) // 2)
    index = 0
    for x, y in pairwise(range(n)):
        distances[index] = np.linalg.norm(grads[x] - grads[y])
        index += 1

    # Select the set of the smallest diameter
    sel_node_set = None
    sel_diam = float('inf')
    for cur_iset in itertools.combinations(range(n), n - f):
        # Compute the current diameter (max of pairwise distances)
        cur_diam = 0.
        for x, y in pairwise(cur_iset):
            # Get distance between these two neighbor_messages
            cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
            # Check if new maximum
            if cur_dist > cur_diam:
                cur_diam = cur_dist
        # Check if new selected diameter
        if cur_diam < sel_diam:
            sel_node_set = cur_iset
            sel_diam = cur_diam

    # Return the selected neighbor_messages
    return sel_node_set

def brute(grads, byzantine_size):
    """ Brute rule.
    Args:
      neighbor_messages Non-empty list of neighbor_messages to aggregate
      byzantine_size  Number of Byzantine neighbor_messages to tolerate
    Returns:
      Aggregated gradient
    """
    sel_node_set = brute_selection(grads, byzantine_size)
    aggregated_gradient = np.mean([grads[i] for i in sel_node_set], axis=0)
    return aggregated_gradient

# Bulyan
def krum_index(grads, byzantine_size):
    n = len(grads)
    k = n - byzantine_size - 2
    scores = np.zeros(n)

    for i in range(n):
        distances = np.array([np.linalg.norm(grads[i] - grads[j]) for j in range(n)])
        scores[i] = np.sum(np.sort(distances)[1:k+1])

    return np.argmin(scores)

def median(grads):
    return np.median(grads, axis=0)

def bulyan(grads, byzantine_size):
    remain = grads
    selected_ls = []
    node_size = len(grads)
    selection_size = node_size - 2 * byzantine_size

    if selection_size <= 0:
        selection_size = 1

    for _ in range(selection_size):
        res_index = krum_index(remain, byzantine_size)
        selected_ls.append(remain[res_index])
        remain = np.delete(remain, res_index, axis=0)

    selection = np.stack(selected_ls)
    m = median(selection)
    dist = -np.abs(selection - m)
    k = max(selection_size - 2 * byzantine_size, 1)
    indices = np.argsort(dist, axis=0)[-k:]

    if len(grads.shape) == 1:
        result = np.mean(selection[indices], axis=0)
    else:
        result = np.array([np.mean(selection[indices[:, d], d], axis=0) for d in range(grads.shape[1])])

    return result

def clip(v, tau = 0.5):
    v_norm = np.linalg.norm(v)
    scale = min(1, tau / v_norm) if v_norm > 0 else 1
    return v * scale

def centered_clipping(inputs, momentum_cc, n_iter = 1):
    n = len(inputs)
    filtered = [v for v in inputs if v is not None]
    momentum_cc = momentum_cc.reshape(filtered[0].shape)
    sum_momentum = np.copy(momentum_cc)

    for _ in range(n_iter):
        for v in filtered:
            sum_momentum += clip(v - momentum_cc)
        momentum_cc = sum_momentum / n

    return np.copy(momentum_cc)

# Sign Guard
def norm_filtering(grads, left_norm= 0.1, right_norm = 3.0):
    all_norm = [np.linalg.norm(mes) for mes in grads]
    median_norm = np.median(all_norm, axis=0)
    S1_idxs = []
    weights = []
    for i, norm in enumerate(all_norm):
        tem = norm / median_norm
        weights.append(min(1,tem))
        if right_norm >= tem >= left_norm:
            S1_idxs.append(i)
    return weights, S1_idxs

def sign_clustering(grads):
    """
    符号聚类算法

    参数:
    neighbor_messages: 二维数组，每行代表一个节点的梯度向量

    返回:
    S2_idxs: 选择的主要聚类簇的节点索引列表
    """
    # 随机选择一部分梯度坐标
    selected_coords = np.random.choice(grads.shape[1], size=min(5, grads.shape[1]), replace=False)

    # 计算选择坐标上的符号统计信息作为特征
    features = []
    for update in grads:
        sign_stats = np.mean(np.sign(update[selected_coords]))
        features.append([sign_stats])  # 转换为二维数组

    # 训练 Mean-Shift 聚类模型
    clustering_model = MeanShift()
    clustering_model.fit(features)

    # 选择具有最多元素的聚类簇作为 S2
    unique_labels, label_counts = np.unique(clustering_model.labels_, return_counts=True)
    main_cluster_label = unique_labels[np.argmax(label_counts)]

    S2_idxs = [idx for idx, label in enumerate(clustering_model.labels_) if label == main_cluster_label]

    return S2_idxs

def sign_guard(grads):
    weights, S1_idxs = norm_filtering(grads)
    S2_idxs = sign_clustering(grads)
    S = [x for x in S1_idxs if x in S2_idxs]
    if S == []:
        S = S1_idxs
    weights = np.array(weights)
    res = np.dot(weights[S], grads[S])/len(S)
    return res

def dnc_agr(grads, num_iters = 5, sub_dim = 100, filter_frac = 1.0, num_byzantine = 2):
    """
    DNC_AGR 函数

    参数:
    grads: 二维数组，每行代表一个节点的梯度向量
    num_iters: 迭代次数
    sub_dim: 子维度大小
    filter_frac: 过滤比例
    num_byzantine: 拟议的拜占庭节点数量

    返回:
    过滤后的梯度均值
    """
    updates = np.array(grads)
    d = updates.shape[1]

    benign_ids = []
    for _ in range(num_iters):
        indices = np.random.permutation(d)[:sub_dim]
        sub_updates = updates[:, indices]
        mu = sub_updates.mean(axis=0)
        centered_update = sub_updates - mu
        _, _, vt = np.linalg.svd(centered_update, full_matrices=False)
        v = vt[0, :]
        s = np.array([(np.dot(update - mu, v) ** 2) for update in sub_updates])

        good = np.argsort(s)[:len(updates) - int(filter_frac * num_byzantine)]
        benign_ids.append(good)

    # 计算多个列表的交集
    intersection_set = set(benign_ids[0])
    for lst in benign_ids[1:]:
        intersection_set.intersection_update(lst)

    benign_ids = list(intersection_set)
    benign_updates = updates[benign_ids, :].mean(axis=0)
    return benign_updates