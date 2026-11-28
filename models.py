# -*- coding: utf-8 -*-
"""
models.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ============================================================
# Device selection
# ============================================================
def get_device():
    """Pick an available device (prefer GPU)"""
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    return device

DEVICE = get_device()  # module-level device variable


# ============================================================
# Time / positional embeddings
# ============================================================
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        # Pre-create the constant tensor and move it to the target device to avoid repeated creation
        self.log_10000 = torch.log(torch.tensor([10000.0])).to(DEVICE)

    def forward(self, x: torch.Tensor):
        # Always use the global DEVICE instead of a hard-coded cuda
        x = x.to(DEVICE) * self.scale
        half_size = self.size // 2

        # Avoid re-creating the arange tensor; keep the device consistent
        emb = self.log_10000 / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(DEVICE))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


# ============================================================
# Legacy diffusion: Unet + NoiseScheduler
# ============================================================
class Unet(nn.Module):
    def __init__(self, emb_size: int = 128, time_emb: str = "sinusoidal",
                 out_size: int = 128):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        concat_size = 2 * emb_size
        layers = []
        layers.append(nn.Linear(concat_size, 2000))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2000, 500))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(500, 500))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(500, 2000))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2000, out_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        # Make sure the input tensors live on the same device
        x = x.to(DEVICE)
        t = t.to(DEVICE)
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 beta_schedule="linear"):
        # FIX: convert num_timesteps to a tensor and move it to the device
        self.num_timesteps = torch.tensor(num_timesteps, dtype=torch.int).to(DEVICE)

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps,
                                        dtype=torch.float32).to(DEVICE)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(DEVICE)
            self.alphas_cumprod_prev = F.pad(
                self.alphas_cumprod[:-1], (1, 0), value=1.).to(DEVICE)

            self.sqrt_alphas_cumprod = (self.alphas_cumprod ** 0.5).to(DEVICE)
            self.sqrt_one_minus_alphas_cumprod = ((1 - self.alphas_cumprod) ** 0.5).to(DEVICE)

            self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(DEVICE)
            self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
                1 / self.alphas_cumprod - 1).to(DEVICE)

            # Pre-compute the coefficients and move them to the device
            self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev)
                                         / (1. - self.alphas_cumprod)).to(DEVICE)
            self.posterior_mean_coef2 = ((1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
                                         / (1. - self.alphas_cumprod)).to(DEVICE)

    def reconstruct_x0(self, x_t, t, noise):
        # Make sure the index tensors live on the same device
        t = t.to(DEVICE)
        s1 = self.sqrt_inv_alphas_cumprod[t].to(x_t.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(x_t.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        t = t.to(DEVICE)
        s1 = self.posterior_mean_coef1[t].to(x_0.device)
        s2 = self.posterior_mean_coef2[t].to(x_t.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        t = t.to(DEVICE)
        if t == 0:
            return 0

        variance = (self.betas[t] * (1. - self.alphas_cumprod_prev[t])
                    / (1. - self.alphas_cumprod[t])).to(t.device)
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep.to(DEVICE)
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        variance = 0
        if t > 0:
            # Create the noise tensor with an explicit device
            noise = torch.randn_like(model_output, device=model_output.device)
            variance = (self.get_variance(t) ** 0.5) * noise
            pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps, device=None):
        # Accept a device argument or fall back to the default DEVICE
        target_device = device if device is not None else DEVICE
        timesteps = timesteps.to(target_device)
        s1 = self.sqrt_alphas_cumprod[timesteps].to(target_device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].to(target_device)
        s1 = s1.view(-1, 1)
        s2 = s2.view(-1, 1)
        out1 = s1 * x_start.to(target_device)
        out2 = s2 * x_noise.to(target_device)
        out = out1 + out2
        return out

    def __len__(self):
        return self.num_timesteps.item()  # convert to a python integer


# ============================================================
# Autoencoder + ClusterProject
# ============================================================
class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self, encoder_dim, activation='relu', batchnorm=True):

        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            # FIX: the original code mistakenly used encoder_layers here
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        x = x.to(DEVICE)
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        latent = latent.to(DEVICE)
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        x = x.to(DEVICE)
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class ClusterProject(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(DEVICE)
        z = self.cluster_projector(x)
        y = self.cluster(z)
        return y, z


# ============================================================
# Orthogonal QR utility (legacy)
# ============================================================
def solve_orthogonal_vector_batch_vectorized_qr(h1, h2):
    B, D = h1.shape

    # Build the matrix [h1, h2, I] in R^{D x (D+2)};
    # we look for a vector orthogonal to both h1 and h2
    eye = torch.eye(D, device=h1.device).unsqueeze(0).expand(B, -1, -1)
    M = torch.cat([h1.unsqueeze(2), h2.unsqueeze(2), eye], dim=2)  # [B, D, D+2]

    # Batched QR decomposition
    Q, R = torch.linalg.qr(M, mode='complete')  # Q: [B, D, D]

    # The columns of Q are vectors orthogonal to the preceding ones;
    # take the last column (QR proceeds column by column)
    h_orth = Q[:, :, -1]  # [B, D]

    return h_orth


# ============================================================
# View attention modules (legacy 2-view family)
# ============================================================
class OrthogonalResidualGenerator(nn.Module):
    """Generate h3 that is orthogonal to both h1 and h2
    (null-space / residual representation)"""

    def __init__(self, latent_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, h1, h2, base):
        """
        h1, h2 : [B, D]
        base   : source feature (can be h1+h2, noise, shared embedding)
        """

        v = self.generator(base)

        h1n = F.normalize(h1, dim=1)
        h2n = F.normalize(h2, dim=1)

        proj1 = (v * h1n).sum(dim=1, keepdim=True) * h1n
        proj2 = (v * h2n).sum(dim=1, keepdim=True) * h2n

        h3 = v - proj1 - proj2
        h3 = F.normalize(h3, dim=1)

        return h3


class AttentionLayer(nn.Module):
    def __init__(self, latent_dim):
        super(AttentionLayer, self).__init__()
        self._latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self._latent_dim * 2, 2, bias=True)

    def forward(self, h1, h2, tau=10.0):
        # Make sure the inputs live on the same device
        h1 = h1.to(DEVICE)
        h2 = h2.to(DEVICE)
        h = torch.cat((h1, h2), dim=1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        e = F.softmax(act, dim=1)
        h = e[:, 0].unsqueeze(1) * h1 + e[:, 1].unsqueeze(1) * h2
        return h


class ViewAttentionDiff(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.orthogonalResidualGenerator = OrthogonalResidualGenerator(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.out = nn.Linear(latent_dim, 2)

    def forward(self, h1, h2, tau=1.0):
        # diff = torch.abs(h1 - h2)
        prod = h1 * h2
        z = self.orthogonalResidualGenerator(h1, h2, prod)
        z = torch.cat([h1, h2, z], dim=1)

        logits = self.out(self.mlp(z))
        e = F.softmax(logits / tau, dim=1)

        h = e[:, 0:1] * h1 + e[:, 1:2] * h2
        return h


class ViewAttentionConfidence(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 4 + 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.out = nn.Linear(latent_dim, 2)

    def forward(self, h1, h2, tau=1.0):
        diff = torch.abs(h1 - h2)
        prod = h1 * h2

        conf1 = torch.norm(h1, dim=1, keepdim=True)
        conf2 = torch.norm(h2, dim=1, keepdim=True)

        z = torch.cat([h1, h2, diff, prod, conf1, conf2], dim=1)

        logits = self.out(self.mlp(z))
        e = F.softmax(logits / tau, dim=1)

        h = e[:, 0:1] * h1 + e[:, 1:2] * h2
        return h


class ViewAttentionCross(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(latent_dim, latent_dim)

    def forward(self, h1, h2, tau=1.0):
        s1 = torch.sum(self.q(h1) * self.k(h2), dim=1, keepdim=True)
        s2 = torch.sum(self.q(h2) * self.k(h1), dim=1, keepdim=True)

        logits = torch.cat([s1, s2], dim=1)
        e = F.softmax(logits / tau, dim=1)

        h = e[:, 0:1] * h1 + e[:, 1:2] * h2
        return h


class ViewAttentionGlobal(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, h1, h2, tau=1.0):
        e = F.softmax(self.weight / tau, dim=0).unsqueeze(0)
        e = e.repeat(h1.size(0), 1)

        h = e[:, 0:1] * h1 + e[:, 1:2] * h2
        return h


class SubspaceOrthogonalResidualGenerator(nn.Module):
    """Generate a residual representation lying in the orthogonal complement
    of the dynamic view subspace span(h1, h2)."""

    def __init__(self, latent_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, h1, h2, base):
        """
        h1, h2 : [B, D]
        base   : shared semantic source (e.g., h1 (.) h2)
        """

        # Step 1: generate a candidate residual
        v = self.generator(base)  # [B, D]

        # Step 2: construct the view subspace
        # H: [B, D, 2]
        H = torch.stack([h1, h2], dim=2)

        # QR decomposition to obtain an orthonormal basis
        # Q: [B, D, 2]
        Q, _ = torch.linalg.qr(H)

        # Step 3: project v onto the view subspace
        v_exp = v.unsqueeze(2)  # [B, D, 1]
        proj = torch.sum(Q * v_exp, dim=1, keepdim=True) * Q
        proj = proj.sum(dim=2)  # [B, D]

        # Step 4: orthogonal residual
        h3 = v - proj
        h3 = F.normalize(h3, dim=1)

        return h3


class ResidualGuidedViewAttention(nn.Module):
    """Subspace-aware orthogonal residual-guided view attention."""

    def __init__(self, latent_dim):
        super().__init__()

        self.residual_generator = SubspaceOrthogonalResidualGenerator(latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.out = nn.Linear(latent_dim, 2)

    def forward(self, h1, h2):
        """
        h1, h2: [B, D]
        """

        # Shared semantic source
        base = h1 * h2

        # Orthogonal residual
        h3 = self.residual_generator(h1, h2, base)

        # Attention prediction
        z = torch.cat([h1, h2, h3], dim=1)
        logits = self.out(self.mlp(z))  # [B, 2]

        # Residual-guided temperature
        # tau = 1.0 + torch.norm(h3, dim=1, keepdim=True)

        # Soft attention
        alpha = F.softmax(logits, dim=1)

        # View fusion
        h = alpha[:, 0:1] * h1 + alpha[:, 1:2] * h2

        return h


# ============================================================
# SPA regression version: cross-view alignment projection head + NT-Xent
# ============================================================
class ProjectionHead(nn.Module):
    """SimCLR projection head 128 -> 128 -> 64; participates only in losses,
    not in the backbone / inference path"""

    def __init__(self, dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, out_dim))

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


class CrossViewNTXent(nn.Module):
    """Same spot across views = positive pair (diagonal), remaining spots in
    the batch = negatives; symmetric NT-Xent"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.t = temperature

    def forward(self, zi, zj):
        # Inputs must already be L2-normalized
        B = zi.shape[0]
        sim = zi @ zj.t() / self.t  # [B, B]
        labels = torch.arange(B, device=zi.device)
        return 0.5 * (F.cross_entropy(sim, labels)
                      + F.cross_entropy(sim.t(), labels))


class MultiViewAttention(nn.Module):
    """Multi-view attention fusion (V-view generalization of the original
    ResidualGuidedViewAttention(z1, z2); if your implementation supports
    multi-input, replace self.AttentionLayer directly)"""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, *latents):
        Z = torch.stack(latents, dim=1)  # [B, V, D]
        w = torch.softmax(self.score(Z).squeeze(-1), dim=1)
        return (Z * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
