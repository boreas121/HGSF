# -*- coding: utf-8 -*-
"""
losses.py
"""
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = sys.float_info.epsilon   # used by ClusterLoss and externally (same as the original util.py)
_EPS_IMC = 1e-8                # used inside IMCClusterLoss (the original cell had EPS = 1e-8)


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""
    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k), "the dimensions of the two views must match"

    # Element-wise pairwise joint probabilities
    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)            # sum over the batch dimension
    p_i_j = (p_i_j + p_i_j.t()) / 2.    # symmetrize so that P = P^T
    p_i_j = p_i_j / p_i_j.sum()         # normalize so the total sums to 1

    return p_i_j


def MMI(view1, view2, lamb=10.0, EPS=sys.float_info.epsilon):
    """MMI loss (mutual-information maximization loss)"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k)), "the joint probability matrix must be of shape (k, k)"

    # Compute the marginal probabilities P_i and P_j
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

    # Guard against log(0)
    p_i_j = torch.clamp(p_i_j, min=EPS)
    p_j = torch.clamp(p_j, min=EPS)
    p_i = torch.clamp(p_i, min=EPS)

    # MMI loss
    loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))
    loss = loss.sum()

    return loss


class ClusterLoss(nn.Module):
    """Cluster-level contrastive loss (with a negative-entropy penalty term)"""

    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num      # number of clusters
        self.temperature = temperature  # temperature coefficient controlling the similarity distribution
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)  # cosine similarity

    def mask_correlated_clusters(self, class_num):
        """Generate the mask matrix that hides positive pairs and self-pairs"""
        N = 2 * class_num
        mask = torch.ones((N, N), device=self.device)
        mask = mask.fill_diagonal_(0)   # hide self-pairs
        for i in range(class_num):
            # hide positive pairs (the two views of the same cluster)
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        """
        Forward pass
        Args:
            c_i, c_j: cluster features of the two views, shape (batch_size, class_num)
            alpha: weight of the negative-entropy loss
        """
        # Negative-entropy loss (encourages a uniform cluster distribution)
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()  # normalize
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + EPS)).sum()  # add EPS to avoid log(0)
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + EPS)).sum()
        ne_loss = ne_i + ne_j

        # Transpose the features -> shape (class_num, batch_size)
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)  # concatenate the features of the two views

        # Cosine similarity divided by the temperature
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        # Similarities of the positive pairs (the two views of the same cluster)
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        # Build the positive/negative pairs
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # with torch.no_grad():
        #     sim_proto = F.cosine_similarity(
        #         c.unsqueeze(1), c.unsqueeze(0), dim=2
        #     )
        #     weight = 1 - sim_proto**5
        #     weight = weight[self.mask].reshape(N, -1)

        negative_clusters = sim[self.mask].reshape(N, -1)
        # negative_clusters = negative_clusters * weight


        # Cross-entropy loss
        labels = torch.zeros(N, device=self.device, dtype=torch.long)
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N  # normalize per sample

        return loss + alpha * ne_loss


class CLoss(nn.Module):
    """Instance-level contrastive loss (fixes the wrong base-class name in the original code)"""

    def __init__(self, batch_size, temperature, device):
        super(CLoss, self).__init__()  # the original code wrongly inherited from InstanceLoss; fixed to CLoss
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        """Generate the mask matrix that hides positive pairs and self-pairs"""
        N = 2 * batch_size
        mask = torch.ones((N, N), device=self.device)
        mask = mask.fill_diagonal_(0)   # hide self-pairs
        for i in range(batch_size):
            # hide positive pairs (the two augmented views of the same instance)
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        """
        Forward pass
        Args:
            z_i, z_j: instance features of the two augmented views, shape (batch_size, feature_dim)
        """
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)  # concatenate the features of the two views

        # Dot-product similarity divided by the temperature
        sim = torch.matmul(z, z.T) / self.temperature
        # Similarities of the positive pairs (the two augmented views of the same instance)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # Build the positive/negative pairs
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # Cross-entropy loss
        labels = torch.zeros(N, device=self.device, dtype=torch.long)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N  # normalize per sample

        return loss


class IMCClusterLoss(nn.Module):
    """
    Incomplete Multi-view Clustering Loss (No Prototype)
    - Cluster-level contrastive
    - Sample-to-cluster alignment
    - Implicit centroid compactness
    - Cross-view centroid consistency
    - Negative entropy (anti-collapse)
    """

    def __init__(self, class_num, temperature, device):
        super().__init__()
        self.K = class_num
        self.T = temperature
        self.device = device

        self.mask = self._mask_correlated_clusters(class_num)
        self.ce = nn.CrossEntropyLoss(reduction="sum")
        self.sim = nn.CosineSimilarity(dim=2)

    # ------------------------------------------------
    # utilities
    # ------------------------------------------------
    def _mask_correlated_clusters(self, K):
        N = 2 * K
        mask = torch.ones((N, N), device=self.device)
        mask.fill_diagonal_(0)
        for k in range(K):
            mask[k, K + k] = 0
            mask[K + k, k] = 0
        return mask.bool()

    # ------------------------------------------------
    # 1. cluster-level contrastive
    # ------------------------------------------------
    def cluster_contrastive(self, c_i, c_j):
        """c_i, c_j: (B, K) soft assignment"""
        c_i = c_i.t()   # (K, B)
        c_j = c_j.t()
        C = torch.cat([c_i, c_j], dim=0)  # (2K, B)

        sim = self.sim(C.unsqueeze(1), C.unsqueeze(0)) / self.T

        pos_i_j = torch.diag(sim, self.K)
        pos_j_i = torch.diag(sim, -self.K)
        positives = torch.cat([pos_i_j, pos_j_i], dim=0).view(2 * self.K, 1)

        negatives = sim[self.mask].view(2 * self.K, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * self.K, device=self.device, dtype=torch.long)

        loss = self.ce(logits, labels) / (2 * self.K)
        return loss

    # ------------------------------------------------
    # 2. sample -> cluster alignment (core of IMC)
    # ------------------------------------------------
    def sample_cluster_alignment(self, c_i, c_j):
        """Encourage the same sample to be assigned to the same cluster across views"""
        c_i = c_i / (c_i.sum(dim=1, keepdim=True) + _EPS_IMC)
        c_j = c_j / (c_j.sum(dim=1, keepdim=True) + _EPS_IMC)

        loss_ij = -(c_i * torch.log(c_j + _EPS_IMC)).sum(dim=1).mean()
        loss_ji = -(c_j * torch.log(c_i + _EPS_IMC)).sum(dim=1).mean()
        return 0.5 * (loss_ij + loss_ji)

    # ------------------------------------------------
    # 3. implicit centroid compactness (guarantees centroids exist)
    # ------------------------------------------------
    def cluster_variance(self, z, q):
        """z: (B, D) embedding ; q: (B, K) soft assignment"""
        mass = q.sum(dim=0) + _EPS_IMC                     # (K,)
        mu = (q.t() @ z) / mass.unsqueeze(1)               # (K, D)

        var = (q.unsqueeze(2)
               * (z.unsqueeze(1) - mu.unsqueeze(0)).pow(2)).sum(dim=(0, 2)) / mass

        return var.mean()

    # ------------------------------------------------
    # 4. cross-view centroid consistency (IMC-specific)
    # ------------------------------------------------
    def centroid_consistency(self, z_i, z_j, c_i, c_j):
        """Align the implicit centroids across views"""
        mass_i = c_i.sum(dim=0) + _EPS_IMC
        mass_j = c_j.sum(dim=0) + _EPS_IMC

        mu_i = (c_i.t() @ z_i) / mass_i.unsqueeze(1)
        mu_j = (c_j.t() @ z_j) / mass_j.unsqueeze(1)

        return F.mse_loss(mu_i, mu_j)

    # ------------------------------------------------
    # 5. negative entropy (anti-collapse)
    # ------------------------------------------------
    def neg_entropy(self, c):
        p = c.sum(0)
        p = p / (p.sum() + _EPS_IMC)
        return math.log(p.size(0)) + (p * torch.log(p + _EPS_IMC)).sum()

    # ------------------------------------------------
    # forward
    # ------------------------------------------------
    def forward(self, c_i, c_j,
                w_contrast=1.0, w_align=1.0, w_var=0.5,
                w_centroid=1.0, w_entropy=0.5):
        L_contrast = self.cluster_contrastive(c_i, c_j)
        # L_align = self.sample_cluster_alignment(c_i, c_j)
        # L_var = self.cluster_variance(z_i, c_i) + self.cluster_variance(z_j, c_j)
        # L_centroid = self.centroid_consistency(z_i, z_j, c_i, c_j)
        L_entropy = self.neg_entropy(c_i) + self.neg_entropy(c_j)

        loss = (w_contrast * L_contrast
                # + w_align * L_align
                # + w_var * L_var
                # + w_centroid * L_centroid
                + w_entropy * L_entropy)

        return loss


def entropy(p):
    return -(p * torch.log(p + EPS)).sum(dim=1, keepdim=True)


def log_center(p, eps=1e-8):
    logp = (p + eps).log()
    return torch.exp(logp - logp.mean(dim=1, keepdim=True))
