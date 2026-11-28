# -*- coding: utf-8 -*-
"""
flow.py 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import PositionalEmbedding


class FlowNet(nn.Module):
    def __init__(self, latent_dim, time_emb=128):
        super().__init__()
        self.time_embed = PositionalEmbedding(time_emb, "sinusoidal")
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_emb, 2056), nn.ReLU(),
            nn.Linear(2056, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 2056), nn.ReLU(),
            nn.Linear(2056, latent_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=1))


class FlowScheduler:
    def __init__(self, num_steps=1, beta_t=0.5, eps=1e-4, symmetric=True,
                 curv=1e-9, ball_eps=1e-4):
        self.num_steps = int(num_steps)
        self.dt = 1.0 / self.num_steps
        self.beta_t = beta_t
        self.eps = eps
        self.symmetric = symmetric
        self._symmetric_trained = self.symmetric
        assert curv > 0, "curv must be positive; as curv -> 0+ the geodesic degenerates into a straight line"
        self.curv = float(curv)
        self.sqrt_c = math.sqrt(self.curv)
        self.ball_eps = float(ball_eps)
        self.euclidean_mode = (self.curv < 1e-8)
        self.max_norm = (1.0 - self.ball_eps) / max(self.sqrt_c, 1e-12)

    def set_curv(self, curv):
        assert curv > 0
        self.curv = float(curv)
        self.sqrt_c = math.sqrt(self.curv)
        self.euclidean_mode = (self.curv < 1e-8)
        self.max_norm = (1.0 - self.ball_eps) / max(self.sqrt_c, 1e-12)

    # ---------- Poincare ball elementary ops ----------
    def _project(self, x):
        if self.euclidean_mode:
            return x
        norm = x.norm(dim=-1, keepdim=True)
        scale = (self.max_norm / norm.clamp_min(self.eps)).clamp(max=1.0)
        return x * scale

    @staticmethod
    def _safe_atanh(x):
        x = x.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _mobius_add(self, x, y):
        if self.euclidean_mode:
            return x + y
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        c = self.curv
        num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
        den = 1.0 + 2.0 * c * xy + c * c * x2 * y2
        return num / den.clamp_min(1e-12)

    def _mobius_scalar_mul(self, r, v):
        if self.euclidean_mode:
            return r * v
        v_norm = v.norm(dim=-1, keepdim=True)
        direction = torch.where(
            v_norm > self.eps,
            v / v_norm.clamp_min(self.eps),
            torch.zeros_like(v))
        arg = (self.sqrt_c * v_norm).clamp(max=1.0 - self.ball_eps)
        small = arg < 1e-4
        atanh_arg = torch.where(
            small, arg + arg ** 3 / 3.0, self._safe_atanh(arg))
        mag = torch.tanh(r * atanh_arg) / self.sqrt_c
        return mag * direction

    def _geodesic(self, x0, x1, t):
        v = self._mobius_add(-x0, x1)
        w = self._mobius_scalar_mul(t, v)
        return self._project(self._mobius_add(x0, w))

    # ---------- Time sampling ----------
    def sample_t(self, batch, device):
        if abs(self.beta_t - 1.0) < 1e-6:
            t = torch.rand(batch, 1, device=device)
        else:
            dist = torch.distributions.Beta(self.beta_t, self.beta_t)
            t = dist.sample((batch, 1)).to(device)
        return t.clamp(self.eps, 1.0 - self.eps)

    # ---------- Path construction ----------
    def make_path(self, x0, x1, cluster_layer=None, return_info=False):
        x0 = self._project(x0)
        x1 = self._project(x1)
        path = lambda t: self._geodesic(x0, x1, t)
        return (path, None) if return_info else path

    def interpolate(self, x0, x1, t, cluster_layer=None, return_info=False):
        x0 = self._project(x0)
        x1 = self._project(x1)
        xt = self._geodesic(x0, x1, t)
        return (xt, None) if return_info else xt

    # ---------- Secant residual loss ----------
    def secant_residual(self, x_a, x_b, u_pred, t_a, t_b):
        w = (t_b - t_a).clamp_min(self.eps)
        return F.mse_loss(w * u_pred, x_b - x_a)

    # ---------- Velocity consistency loss (replaced by L_cdc; kept for reference) ----------
    def velocity_consistency_loss(self, model, x_t1, x_t2, t1, t2):
        v1 = model(x_t1, t1.squeeze(-1))
        v2 = model(x_t2, t2.squeeze(-1))
        lhs = (1 - t2) * v2 - (1 - t1) * v1
        rhs = x_t1 - x_t2
        return F.mse_loss(lhs, rhs)

    def velocity_consistency_loss_grid(self, model, path, t, dt):
        t = t.clamp(max=max(1.0 - 2.0 * dt, 0.0))
        x_a = path(t)
        x_m = path(t + dt)
        x_c = path(t + 2.0 * dt)
        v_a = model(x_a, t.squeeze(-1))
        v_m = model(x_m, (t + dt).squeeze(-1))
        return F.mse_loss(dt * (v_a + v_m), x_c - x_a)

    def velocity_consistency(self, model, path, B, device, dt_min=0.3, dt_max=0.5):
        if self.num_steps == 1:
            t1 = torch.rand(B, 1, device=device) * (0.5 - dt_min) + dt_min * 0.5
            dt12 = torch.rand(B, 1, device=device) * (dt_max - dt_min) + dt_min
            t2 = (t1 + dt12).clamp(max=1.0 - self.eps)
            return self.velocity_consistency_loss(model, path(t1), path(t2), t1, t2)
        t = self.sample_t(B, device).clamp(max=max(1.0 - 2.0 * self.dt, self.eps))
        return self.velocity_consistency_loss_grid(model, path, t, self.dt)

    # ---------- NEW: cumulative displacement consistency (paper L_cdc: eqn:cum / eqn:Phi / eqn:cdc) ----------
    def _cum_endpoint(self, model, path, t, h):
        """Phi_theta(t) = gamma*(t) + sum_{k=0}^{M(t)-1} w_k(t) * F_theta(gamma*(t+kh), t+kh),
        where w_k(t) = h for k < M(t)-1 and w_{M(t)-1}(t) = r(t) = ell(t) - (M(t)-1)h, with ell(t) = 1-t"""
        ell = 1.0 - t                                          # ell(t), [B,1]
        M = torch.clamp(torch.ceil(ell / h - 1e-8), min=1.0)   # M(t) = max{1, ceil(ell/h)}
        r = (ell - (M - 1.0) * h).clamp_min(0.0)               # residual length of the last step r(t)
        max_M = int(M.max().item())
        x_t = path(t)
        cum = torch.zeros_like(x_t)
        for k in range(max_M):
            tk = (t + k * h).clamp(0.0, 1.0)
            wk = torch.where(k < M - 1.0, torch.full_like(t, h), r)  # w_k(t), [B,1]
            valid = (k <= M - 1.0).to(x_t.dtype)                     # mask k > M(t)-1
            v = model(path(tk), tk.squeeze(-1))                      # F_theta(gamma*(t+kh), t+kh)
            cum = cum + wk * v * valid
        return x_t + cum

    def cdc_consistency(self, model, path, B, device, num_steps=10):
        """L_cdc = E_{t1,t2~U[0,1], t1<t2} || Phi_theta(t1) - Phi_theta(t2) ||^2:
        the cumulative predicted displacements starting from different points along
        the geodesic must converge to the same endpoint (closed vector triangle)"""
        h = 1.0 / float(num_steps)
        t1 = torch.rand(B, 1, device=device)
        t2 = torch.rand(B, 1, device=device)
        lo = torch.minimum(t1, t2)  # conditional sampling t1 < t2
        hi = torch.maximum(t1, t2)
        return F.mse_loss(self._cum_endpoint(model, path, lo, h),
                          self._cum_endpoint(model, path, hi, h))

    # ---------- One-step extrapolation ----------
    def one_step(self, model, x):
        if self.num_steps != 1:
            raise RuntimeError("one_step is only available when num_steps == 1; use integrate() instead")
        t0 = torch.zeros(x.shape[0], device=x.device)
        return self._project(x + model(x, t0))

    # ---------- Integration ----------
    def integrate(self, model, x, reverse=False):
        if reverse and not self._symmetric_trained:
            raise NotImplementedError(
                "reverse imputation relies on symmetric augmentation (symmetric=True). "
                "With asymmetric training, F is unsupervised near t -> 1 and outside the "
                "trajectory; for a fair asymmetric ablation, train a separate reverse network.")
        if self.num_steps == 1:
            return self.one_step(model, self._project(x))
        device = x.device
        dt = self.dt
        x_t = self._project(x)
        ts = torch.arange(self.num_steps, device=device).to(x.dtype) / self.num_steps
        t_all = ts.view(-1, 1, 1).expand(-1, x.shape[0], 1).contiguous()
        for i in range(self.num_steps):
            t_b = t_all[i]
            v = model(x_t, t_b.squeeze(-1))
            x_t = self._project(x_t + dt * v)
        return x_t
