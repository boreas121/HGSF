# -*- coding: utf-8 -*-
"""
utils.py -- Common utilities.
"""
import sys
import math
import random
import warnings

import numpy as np
import torch
import sklearn.metrics as metrics
from munkres import Munkres
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import UndefinedMetricWarning

EPS = sys.float_info.epsilon


# ============================================================
# Seeding
# ============================================================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================
# Missing-value masks
# ============================================================
def get_mask(view_num, n, missing_rate, seed=None):
    """SPA version: independent Bernoulli missingness per view; ensures each
    row keeps at least one observed view (used by train.py)"""
    rng = np.random.RandomState(seed)
    mask = (rng.rand(n, view_num) > missing_rate).astype('int64')
    empty = mask.sum(1) == 0
    mask[empty, 0] = 1
    return mask


def get_mask_icdm(view_num, data_len, missing_rate):
    """ICDM legacy version: indicator matrix A that iteratively approaches the exact missing rate"""
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(
            np.random.randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = np.random.randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(
            np.random.randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (np.random.randint(0, 100, size=(data_len, view_num))
                       < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (np.random.randint(0, 100, size=(data_len, view_num))
                       < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix  # indicator matrix A


# ============================================================
# Batch iteration / normalization
# ============================================================
def next_batch(X1, X2, X3, X4, batch_size):
    """Return the data for the next batch (legacy 4-view)"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, batch_x3, batch_x4, (i + 1))


def next_batch_v(*args, batch_size):
    """V-view version isomorphic to the original util.next_batch:
    yield (batch_view_0..V-1, flag_0..V-1, batch_no)"""
    n = args[0].shape[0]
    order = np.random.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        out = [a[idx] for a in args]
        yield tuple(out) + (start // batch_size,)


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def target_l2(q):
    return ((q ** 2).t() / (q ** 2).sum(1)).t()


# ============================================================
# File / tensor IO
# ============================================================
def read_txt_list(path):
    arr = np.genfromtxt(path, dtype='str')
    if arr.ndim == 0:
        return [str(arr)]
    return [str(a) for a in arr]


def torch_load_safe(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def to_np(t):
    return t.detach().cpu().float().numpy() if torch.is_tensor(t) \
        else np.asarray(t, dtype='float32')


# ============================================================
# Clustering evaluation metrics (legacy)
# ============================================================
def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision: zero_division=1 silences the warning and sets 1.0 when there is no predicted sample
    precision = metrics.precision_score(y_true, y_pred, average=average, zero_division=1)
    precision = np.round(precision, decimals)

    # recall: zero_division=1 silences the warning
    recall = metrics.recall_score(y_true, y_pred, average=average, zero_division=1)
    recall = np.round(recall, decimals)

    # F-score: zero_division=1 silences the warning
    f_score = metrics.f1_score(y_true, y_pred, average=average, zero_division=1)
    f_score = np.round(f_score, decimals)

    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f_measure': f_score}, confusion_matrix


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i, j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    # explicitly specify the label range of the confusion matrix to avoid empty labels
    labels = range(n_clusters)
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=labels)

    # compute accuracy based on the optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)

    # guard against index out-of-range (extreme-case protection)
    cluster_assignments = np.clip(cluster_assignments, 0, n_clusters - 1)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    # validate n_clusters
    if n_clusters <= 0:
        warnings.warn("n_clusters must be positive, setting to 1", UserWarning)
        n_clusters = 1

    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari},
                **classification_metrics), confusion_matrix


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    # if cluster_obj is None, both ClusterClass and n_clusters are required
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
    for _ in range(10):
        try:
            cluster_obj.fit(x)
            break
        except Exception:
            print("Unexpected error:", sys.exc_info())
    else:
        return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def evaluation(y_pred, y_true, accumulated_metrics=None):
    """Clustering evaluation entry (handles y_true labels starting from 1 / single-label warning)"""
    unique_labels = np.unique(y_true)
    n_clusters = np.size(unique_labels)

    if n_clusters == 0:
        raise ValueError("y_true is empty!")
    if n_clusters == 1:
        warnings.warn("Only one unique label in y_true, clustering metrics may be meaningless",
                      UserWarning)

    if np.min(y_true) == 1:
        y_true = y_true - 1

    scores, _ = clustering_metric(y_true, y_pred, n_clusters)

    if accumulated_metrics is not None:
        # tolerate both key conventions (the original code used uppercase ARI and lowercase others)
        accumulated_metrics['acc'].append(scores['accuracy'])
        accumulated_metrics['nmi'].append(scores['NMI'])
        accumulated_metrics['ARI'].append(scores['ARI'])
        accumulated_metrics['f-mea'].append(scores['f_measure'])

    return scores
