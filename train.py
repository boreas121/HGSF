# -*- coding: utf-8 -*-
"""
train.py 
"""
import os

from config import CFG, DATASETS, make_cfg

# CUDA_VISIBLE_DEVICES must be set before importing torch (original args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = getattr(CFG, 'gpu', '0')

import math
import time
import copy
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    import setproctitle
    # show up as plain python in nvidia-smi
    setproctitle.setproctitle("python")
except ImportError:
    pass

from models import (Autoencoder, ClusterProject, ProjectionHead,
                    CrossViewNTXent, MultiViewAttention)
from flow import FlowNet, FlowScheduler
from losses import ClusterLoss, entropy, log_center
from utils import EPS, get_mask, set_seed, next_batch_v, target_l2
from data import load_spa_data
from metrics import (per_gene_pearson, compute_metrics, log_metrics,
                     aggregate_stats, fmt_metrics, save_metrics_csv,
                     print_final_summary)
from visualization import (plot_scatter_pred_true, plot_hist, plot_top_genes,
                           plot_spatial_maps, plot_rvd_pcc,
                           plot_training_curves, plot_metric_comparison)

warnings.filterwarnings("ignore")

args = CFG()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: " + str(device))

# loss_hist adds the 'rad' (L_rad) and 'comp' (post-completion semantic consistency) monitoring keys
loss_hist = {k: [] for k in ['rec', 'flow', 'vel', 'cluster',
                             'entropy', 'rad', 'comp', 'total']}
val_pcc_list = []


# ============================================================
# Main model: HGSF (V-view regression generalization of the original ICDM)
# ============================================================
class HGSF(nn.Module):
    def __init__(self, config, cfg):
        super().__init__()
        self._config = config
        self.cfg = cfg
        self.view_num = config['view']
        self._latent_dim = config['Autoencoder']['arch1'][-1]

        # ---- Multiple encoders (key names follow the original arch1/arch2/arch3, 1-indexed) ----
        for i in range(1, self.view_num + 1):
            self.add_module(f'autoencoder{i}', Autoencoder(
                config['Autoencoder'][f'arch{i}'],
                config['Autoencoder'][f'activations{i}'],
                config['Autoencoder']['batchnorm']))

        self.flow = FlowNet(self._latent_dim)
        fc = config.get('flow', {})
        self.flow_scheduler = FlowScheduler(
            num_steps=fc.get('num_steps', 1),
            beta_t=fc.get('beta_t', 0.5),
            eps=fc.get('eps', 1e-4),
            symmetric=fc.get('symmetric', True),
            curv=fc.get('curv', 1e-9),
            ball_eps=fc.get('ball_eps', 1e-4),
        )
        # NEW: auto calibration
        self._auto_curv = fc.get('auto_curv', True)
        self._auto_curv_target = fc.get('auto_curv_target', 0.9)
        ace = fc.get('auto_curv_epochs', [3, 30])
        self._auto_curv_epochs = [ace] if isinstance(ace, int) else list(ace)
        self._auto_curv_done = set()
        if (not self._auto_curv) and self.flow_scheduler.euclidean_mode:
            print("[warn] auto_curv=False and curv~=0: paths are pure Euclidean straight lines. "
                  "For a fixed-hyperbolic ablation, set curv explicitly in config['flow'].")
        self._warned_no_sym = False

        self.clusterLayer = ClusterProject(
            self._latent_dim, config['training']['n_clusters'])
        self.AttentionLayer = MultiViewAttention(self._latent_dim)

        # ---- Cross-view alignment (regression version), replaces ClusterLoss(y_i, y_j) ----
        self.align_proj = ProjectionHead(self._latent_dim, out_dim=64)
        self.align_loss = CrossViewNTXent(temperature=0.5)

    def to_device(self, device):
        self.to(device)

    # ================= Utility: normalize the mask dtype =================
    @staticmethod
    def _bool_mask(mask):
        m = torch.as_tensor(mask)
        if m.dtype != torch.bool:
            m = m != 0
        return m

    # ================= Utility: normalize the entropy-weight shape =================
    @staticmethod
    def _entropy_weight(y):
        e = torch.as_tensor(entropy(y), device=y.device)
        if e.dim() == 1:
            e = e.view(-1, 1)
        return torch.exp(-e)

    def _warn_no_sym(self):
        if not self._warned_no_sym:
            self._warned_no_sym = True
            print("[warn] symmetric=False: only the (view0->view1) direction is supervised; "
                  "missing views fall back to cross-view copying (consistent between training "
                  "and evaluation). For a fair asymmetric ablation, consider training a reverse "
                  "FlowNet and adding its parameters to the optimizer.")

    def _supervised_pair(self, u, v):
        """Asymmetric mode: only (0 -> 1) is supervised (matching the original z1 -> z2);
        symmetric mode: any ordered pair"""
        if self.flow_scheduler.symmetric:
            return True
        return (u == 0 and v == 1)

    # ================= NEW: auto-calibrate the curvature (V-view generalization) =================
    @torch.no_grad()
    def _auto_calibrate_curv(self, X_dev, mask_b, device, safety=1.15, n_sample=512):
        """Calibrate curv from the 99.9% norm quantile of the truly observed samples of
        every view (original logic generalized to V views)"""
        for i in range(1, self.view_num + 1):
            getattr(self, f'autoencoder{i}').eval()
        rs = []
        for v in range(self.view_num):
            idx = mask_b[:, v].nonzero(as_tuple=True)[0].cpu()
            if len(idx) == 0:
                continue
            sel = idx[torch.randperm(len(idx))[:n_sample]]
            x = X_dev[v]
            xb = x[sel.to(x.device)].to(device=device, dtype=torch.float32)
            rs.append(getattr(self, f'autoencoder{v+1}').encoder(xb)
                      .norm(dim=-1).cpu())
        for i in range(1, self.view_num + 1):
            getattr(self, f'autoencoder{i}').train()
        if not rs:
            return
        R_z = torch.quantile(torch.cat(rs), 0.999).item()
        if (not math.isfinite(R_z)) or R_z < 1e-6:
            return
        curv_new = (self._auto_curv_target / (safety * R_z)) ** 2
        self.flow_scheduler.set_curv(curv_new)
        print(f"[auto_curv] R_z(99.9%)={R_z:.4f}, curv->{curv_new:.4f}, "
              f"R_ball={1.0 / math.sqrt(curv_new):.4f}, "
              f"approx. ball share={self._auto_curv_target / safety:.2f}")

    # ================= Training (V-view generalization of the original train) =================
    def train_poinc(self, config, X_dev, mask_dev, Y, tr_idx, val_idx,
                    optimizer, device):
        criterion_cluster = ClusterLoss(  # kept from the original implementation (currently unused)
            config['training']['n_clusters'], 1.0, device).to(device)
        mask_b = self._bool_mask(mask_dev)
        flag = mask_b.long().to(device)
        V = self.view_num

        L_1 = self.cfg.lamda_List[self.cfg.flagwithAblation][0]
        L_2 = self.cfg.lamda_List[self.cfg.flagwithAblation][1]
        L_3 = self.cfg.lamda_List[self.cfg.flagwithAblation][2]
        L_4 = self.cfg.lamda_List[self.cfg.flagwithAblation][3]
        L_5 = self.cfg.lamda_List[self.cfg.flagwithAblation][4]  # NEW: lambda_r, weight of L_rad
        L_6 = self.cfg.lamda_List[self.cfg.flagwithAblation][5]  # NEW: lambda, weight of L_comp

        tr_cfg = config['training']
        w_cfg = tr_cfg.get('loss_weights', {})
        n_epochs = tr_cfg.get('n_epochs', 200)
        eval_every = tr_cfg.get('eval_every', 1)

        VEL_WARMUP_EPOCH = w_cfg.get('vel_warmup_epoch', 20)
        VEL_DT_MIN = w_cfg.get('vel_dt_min', 0.3)   # no longer used after CDC; kept for compatibility
        VEL_DT_MAX = w_cfg.get('vel_dt_max', 0.5)   # ditto
        VEL_BOOTSTRAP = w_cfg.get('vel_bootstrap_missing', True)
        CDC_NUM_STEPS = w_cfg.get('cdc_num_steps', 10)  # NEW: L_cdc steps (step=10)

        sched = self.flow_scheduler
        sched._symmetric_trained = sched.symmetric
        best_r, best_state = -np.inf, None
        train_start_time = time.time()
        last_loss = 0.0

        def _safe_log(x):
            """Safe log for the outputs of log_center / target_l2; negative/0 -> 1e-12."""
            return torch.log(torch.clamp(x, min=1e-12))

        for epoch in range(n_epochs):
            # NEW: multi-point calibration
            if self._auto_curv and epoch in self._auto_curv_epochs \
                    and epoch not in self._auto_curv_done:
                self._auto_calibrate_curv(X_dev, mask_b, device)
                self._auto_curv_done.add(epoch)

            # V-view version of the original shuffle(x1, x2, flag1, flag2)
            shuffled = shuffle(*X_dev, *[flag[:, v] for v in range(V)])
            views_s, flags_s = shuffled[:V], shuffled[V:]

            epoch_vel_loss, epoch_n_batch = 0.0, 0
            epoch_s_max, epoch_over, epoch_cnt = 0.0, 0, 0

            for batch in next_batch_v(*views_s, *flags_s,
                                      batch_size=tr_cfg['batch_size']):
                batch_view, x_flags = batch[:V], batch[V:2 * V]
                if len(batch_view[0]) == 1:
                    continue
                index_both = torch.stack(x_flags, 0).sum(0) == V  # complete samples
                if int(index_both.sum()) < 2:
                    continue

                z_list = [getattr(self, f'autoencoder{i+1}').encoder(
                    batch_view[i][index_both]) for i in range(V)]

                # ============ 1. Reconstruction ============
                rec_loss = 0
                for i in range(V):
                    rec_loss = rec_loss + F.mse_loss(
                        getattr(self, f'autoencoder{i+1}').decoder(z_list[i]),
                        batch_view[i][index_both])

                # ============ 2. Flow (geodesics + symmetric augmentation) ============
                # NEW: V-view generalization of the original 50% swap; with symmetry on,
                # each batch draws one random ordered view pair; the asymmetric mode fixes
                # (view0 -> view1) (matching the original z1 -> z2)
                B = z_list[0].shape[0]
                dt = sched.dt
                if sched.symmetric:
                    perm = torch.randperm(V)
                    i_src, j_tgt = int(perm[0]), int(perm[1])
                else:
                    i_src, j_tgt = 0, 1
                src, tgt = z_list[i_src], z_list[j_tgt]
                path = sched.make_path(src, tgt)

                t_a = sched.sample_t(B, device)
                t_b = (t_a + dt).clamp(max=1.0)
                x_a, x_b = path(t_a), path(t_b)
                u_p = self.flow(x_a, t_a.squeeze(-1))
                flow_loss = sched.secant_residual(x_a, x_b, u_p, t_a, t_b)

                # ============ 3. Cumulative Displacement Consistency (CDC) ============
                # NEW: velocity_consistency -> cdc_consistency (paper L_cdc, step=10);
                # the warmup / bootstrap / weight L_2 structure is unchanged
                use_vel = (L_2 > 0) and (epoch >= VEL_WARMUP_EPOCH)
                if use_vel:
                    vel_loss = sched.cdc_consistency(
                        self.flow, path, B, device, CDC_NUM_STEPS)
                    if VEL_BOOTSTRAP:
                        def _bootstrap(z_obs):
                            if z_obs.shape[0] < 2:
                                return torch.tensor(0.0, device=device)
                            with torch.no_grad():
                                z_pseudo = sched.integrate(
                                    self.flow, sched._project(z_obs))
                            # NEW: pseudo path with the same geodesic construction as the main path (unchanged)
                            pseudo_path = sched.make_path(z_obs, z_pseudo)
                            return sched.cdc_consistency(
                                self.flow, pseudo_path, z_obs.shape[0], device,
                                CDC_NUM_STEPS)
                        # Original: z1_obs always bootstrapped, z2_obs only when symmetric;
                        # V views: view0 always bootstrapped, other views only when symmetric
                        for vi in range(V):
                            if sched.symmetric or vi == 0:
                                z_obs = getattr(self, f'autoencoder{vi+1}').encoder(
                                    batch_view[vi][x_flags[vi] == 1])
                                vel_loss = vel_loss + _bootstrap(z_obs)
                else:
                    vel_loss = torch.tensor(0.0, device=device)

                h = self.AttentionLayer(*z_list)

                # ============ 4a. Cross-view Consistency (NT-Xent) ============
                # Reuse a single y_ntxent to avoid repeated self.align_proj forwards inside the loop
                y_ntxent = [self.align_proj(z) for z in z_list]
                consis_loss = 0
                for i in range(V):
                    j = (i + 1) % V
                    consis_loss = consis_loss + self.align_loss(
                        y_ntxent[i], y_ntxent[j])
                consis_loss = consis_loss / V

                # ============ 4b. Entropy-weighted hierarchical loss ============
                # Use clusterLayer (softmax probabilities, non-negative) rather than
                # align_proj (signed, L2-normalized); the latter would produce
                # log(negative) = NaN and poison the encoder weights.
                y_list_ent = [self.clusterLayer(z)[0] for z in z_list]
                yh_ent = self.clusterLayer(h)[0]
                y_lgs = [_safe_log(log_center(y) + EPS) for y in y_list_ent]
                yh_lg = _safe_log(log_center(yh_ent) + EPS)
                ws = [self._entropy_weight(y) for y in y_list_ent] + \
                     [self._entropy_weight(yh_ent)]
                w_sum = sum(ws) + EPS
                log_y_star = sum((w / w_sum) * lg for w, lg in zip(ws, y_lgs + [yh_lg]))
                # clamp before exp: defend against occasional positive overflow of log_center
                y_star = target_l2(torch.exp(torch.clamp(log_y_star, max=20.0)))
                entropy_loss = F.kl_div(
                    _safe_log(yh_ent), y_star.detach(), reduction='batchmean')

                # ============ 4c. Geometry consistency (L_rad radius consistency; NEW) ============
                # r_v = ||z_v||_2 (paper eqn:rad); pairs follow the adjacent-ring pairing;
                # reduces to the paper formula with 2 views
                rad_loss = 0
                for i in range(V):
                    j = (i + 1) % V
                    rad_loss = rad_loss + F.mse_loss(
                        z_list[i].norm(dim=-1), z_list[j].norm(dim=-1))
                rad_loss = rad_loss / V

                # ============ 4d. Post-completion semantic consistency (L_comp; NEW, tiny weight) ============
                # Paper's L_clu completion branch: F({y_src},{y_tgt}), y_v = h(z_v), z_v = G(z_u);
                # z is obtained via flow integrate and is NOT detached -- gradients propagate back
                # through integrate into the secant field (paper: "propagates supervision back to
                # the secant field"); reuses the same ordered pair (i_src -> j_tgt) drawn in the
                # flow block; with 2 views this coincides exactly with the paper formulation;
                # the semantic encoding reuses the same align_proj + NT-Xent as L_3 (the
                # implementation vehicle of F in code).
                # Note: with symmetric=False the reverse direction (j -> i) is unsupervised;
                if sched.symmetric:
                    z_hat_tgt = sched.integrate(self.flow, z_list[i_src])  # z_tgt = G(z_src)
                    z_hat_src = sched.integrate(self.flow, z_list[j_tgt])  # z_src = G(z_tgt)
                    comp_loss = self.align_loss(
                        self.align_proj(z_hat_src), self.align_proj(z_hat_tgt))
                else:
                    comp_loss = torch.tensor(0.0, device=device)

                # ============ 5. Total loss (original formula + L_rad * lambda_r + L_comp * lambda) ============
                loss = (rec_loss + flow_loss * L_1 + vel_loss * L_2
                        + consis_loss * L_3 + entropy_loss * L_4
                        + rad_loss * L_5 + comp_loss * L_6)

                # NEW NaN guard: stop-loss in place for any numerically unexpected branch,
                # so gradients do not spread
                if not torch.isfinite(loss):
                    print(f"[warn] ep{epoch}: non-finite loss "
                          f"(rec={float(rec_loss):.3e}, "
                          f"flow={float(flow_loss):.3e}, "
                          f"vel={float(vel_loss):.3e}, "
                          f"consis={float(consis_loss):.3e}, "
                          f"entropy={float(entropy_loss):.3e}, "
                          f"rad={float(rad_loss):.3e}, "
                          f"comp={float(comp_loss):.3e}) -- skip step")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                # check the gradients again: if the loss above was finite but some branch
                # back-propagated inf/nan
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=1e9)  # detect only, no clipping
                if not torch.isfinite(grad_norm):
                    print(f"[warn] ep{epoch}: non-finite grad "
                          f"(||g||={float(grad_norm):.3e}) -- skip step")
                    optimizer.zero_grad()
                    continue
                optimizer.step()

                last_loss = loss.item()
                epoch_vel_loss += vel_loss.item()
                epoch_n_batch += 1

                # NEW: epoch-wise in-ball monitoring (covering all views)
                with torch.no_grad():
                    s_b = torch.cat([z.norm(dim=-1) for z in z_list]) \
                        * sched.sqrt_c
                    epoch_s_max = max(epoch_s_max, s_b.max().item())
                    epoch_over += (s_b > 1.0 - sched.ball_eps).sum().item()
                    epoch_cnt += s_b.numel()

                # ---- loss_hist logging (empty epochs explicitly filled with 0, no longer
                #      reading undefined variables) ----
                if epoch_n_batch > 0:
                    batch_losses = {
                        'rec': rec_loss,
                        'flow': flow_loss,
                        'vel': vel_loss,         # semantics now L_cdc (cumulative displacement consistency)
                        'cluster': consis_loss,  # semantics of term 4: cross-view NT-Xent
                        'entropy': entropy_loss,
                        'rad': rad_loss,         # NEW: L_rad
                        'comp': comp_loss,       # NEW: post-completion semantic consistency L_comp
                        'total': loss,
                    }
                    for k in loss_hist:
                        loss_hist[k].append(batch_losses[k].item())
                else:
                    for k in loss_hist:
                        loss_hist[k].append(0.0)

            # ==================== Monitoring + validation ====================
            is_calib = (epoch in self._auto_curv_done)
            if (epoch_n_batch > 0 and not sched.euclidean_mode
                    and ((epoch % 10 == 0 and epoch > 0) or is_calib)):
                print(f"[ball] ep{epoch}: max sqrt(c)*||z||={epoch_s_max:.3f}, "
                      f"out-of-ball={epoch_over / max(epoch_cnt, 1):.2%}, "
                      f"curv={sched.curv:.4f}")

            if (epoch % eval_every == 0) or ((epoch + 1) == n_epochs):
                avg_vel = epoch_vel_loss / max(epoch_n_batch, 1)

                # ---- Validation: fused features + Ridge -> val PCC (replaces the original
                #      clustering evaluation) ----
                with torch.no_grad():
                    Z = self.infer_features(X_dev, mask_dev, device)
                if not np.isfinite(Z).all():
                    n_bad = int((~np.isfinite(Z)).sum())
                    print(f"[warn] ep{epoch}: non-finite Z "
                          f"({n_bad}/{Z.size} bad) -- skip validation")
                    val_pcc_list.append(-1.0)
                    continue

                reg = make_pipeline(StandardScaler(),
                                    Ridge(alpha=self.cfg.reg_alpha))
                complete = (mask_dev == 1).all(1).cpu().numpy()
                fit_idx = tr_idx[complete[tr_idx]]
                reg.fit(Z[fit_idx], Y[fit_idx])
                pred_val = reg.predict(Z[val_idx])
                r = float(np.nanmean(per_gene_pearson(pred_val, Y[val_idx])))
                val_pcc_list.append(r)
                if r > best_r:
                    best_r, best_state = r, copy.deepcopy(self.state_dict())

                print(f"Epoch: {epoch}/{n_epochs} ==> loss = {last_loss:.4f}"
                      f" | vel = {avg_vel:.4e} | val PCC = {r:.4f}"
                      f" (best {best_r:.4f})")

        if best_state is not None:
            self.load_state_dict(best_state)
        train_total_time = time.time() - train_start_time
        print("\n" + "=" * 50)
        print(f"Training Finished! Total Epochs: {n_epochs}")
        print(f"Total Training Time: {train_total_time:.2f} s "
              f"({train_total_time / 60:.2f} min)")
        print(f"Best val PCC: {best_r:.4f}")
        print("=" * 50 + "\n")
        return train_total_time, best_r

    # ============ Inference: V-view generalization of the original evaluation / imputation logic ============
    @torch.no_grad()
    def infer_features(self, X_dev, mask_dev, device):
        """
        Original logic: observed latents are projected into the ball; missing v2 <- integrate(latent1);
        missing v1 <- integrate(latent2, reverse=True) (symmetric) or copy (asymmetric).
        V-view generalization: for a missing view v, for each present view u: if the pair (u, v) is
        supervised -> integrate(H[u]); otherwise copy H[u] (fallback); multiple sources are averaged
        row-wise; observed latents are also projected into the ball.
        """
        was_train = self.training
        self.eval()
        for i in range(1, self.view_num + 1):
            getattr(self, f'autoencoder{i}').eval()
        self.flow.eval()

        mask_b = self._bool_mask(mask_dev)
        V, N = self.view_num, X_dev[0].shape[0]
        L = self._latent_dim
        H = torch.zeros(V, N, L, device=device)

        # Observed latents (projected, aligned with the training scale)
        for v in range(V):
            obs = mask_b[:, v]
            if int(obs.sum()) > 0:
                H[v][obs] = self.flow_scheduler._project(
                    getattr(self, f'autoencoder{v+1}').encoder(
                        X_dev[v][obs.to(X_dev[v].device)]))

        # Impute the missing views with the flow
        for v in range(V):
            miss = ~mask_b[:, v]
            if int(miss.sum()) == 0:
                continue
            acc = torch.zeros(N, L, device=device)
            cnt = torch.zeros(N, device=device)
            for u in range(V):
                if u == v:
                    continue
                rows = miss & mask_b[:, u]
                if int(rows.sum()) == 0:
                    continue
                if self._supervised_pair(u, v):
                    z_pred = self.flow_scheduler.integrate(self.flow, H[u][rows])
                else:
                    self._warn_no_sym()
                    z_pred = H[u][rows].clone()  # cross-view copy fallback (original semantics)
                acc[rows] = acc[rows] + z_pred
                cnt[rows] = cnt[rows] + 1
            rows_any = miss & (cnt > 0)
            if int(rows_any.sum()) > 0:
                H[v][rows_any] = acc[rows_any] / cnt[rows_any].unsqueeze(1)

        fused = self.AttentionLayer(*[H[v] for v in range(V)])

        if was_train:
            self.train()
            for i in range(1, self.view_num + 1):
                getattr(self, f'autoencoder{i}').train()
            self.flow.train()
        return fused.cpu().numpy()  # (N, latent_dim)

    # ============ Diagnostics: cluster drift curve (unchanged, pairwise diagnostics) ============
    @torch.no_grad()
    def cluster_drift_curve(self, z0, z1, T=21):
        ts = torch.linspace(0, 1, T, device=z0.device)
        path = self.flow_scheduler.make_path(z0, z1)
        gammas = [path(tt.view(1, 1)) for tt in ts]
        qs = [self.clusterLayer(g)[0] for g in gammas]
        q = torch.stack(qs, dim=1)
        q0 = q[:, 0:1].clamp_min(1e-12)
        kl = (q.clamp_min(1e-12) * (q.clamp_min(1e-12) / q0).log()).sum(-1)
        return kl.mean(0).cpu()

    # ============ FLOPs / parameter profiling (V-view adapted) ============
    def count_flops_params(self, X_dev, mask, config, device):
        try:
            from thop import profile, clever_format
        except ImportError:
            print("warn: thop is not installed; skip the FLOPs profiling")
            return None
        mask_b = self._bool_mask(mask)
        V, N = self.view_num, X_dev[0].shape[0]
        total_flops = 0
        with torch.no_grad():
            z_list = []
            for v in range(V):
                x_v = X_dev[v]
                obs = mask_b[:, v].to(x_v.device)
                if obs.any():
                    flops, _ = profile(getattr(self, f'autoencoder{v+1}'),
                                       inputs=(x_v[obs],))
                    total_flops += flops
                    z_list.append(getattr(self, f'autoencoder{v+1}').encoder(x_v))
                else:
                    z_list.append(torch.zeros(N, self._latent_dim, device=device))
            # attention: thop does not support variable-length lists; approximate with the first two latents
            flops_att, _ = profile(self.AttentionLayer,
                                   inputs=(z_list[0], z_list[1]))
            h = self.AttentionLayer(*z_list)
            flops_cluster, _ = profile(self.clusterLayer, inputs=(h,))
            t = torch.rand(N, device=device)
            flops_flow, _ = profile(self.flow, inputs=(z_list[0], t))
            flops_flow *= self.flow_scheduler.num_steps  # the imputation runs N forward steps
            mr = config['training'].get('missing_rate', 0.5)
            total_flops += flops_att + flops_cluster + flops_flow * V * mr
        total_flops = total_flops / N
        flops_str = clever_format([total_flops], "%.3f")
        print(f"Total FLOPs (per sample, V-view approximation): {flops_str}")
        return total_flops

    def measure_inference_time(self, X_dev, mask_dev, device, repeat=20):
        times = []
        with torch.no_grad():
            for _ in range(3):
                _ = self.infer_features(X_dev, mask_dev, device)
            for _ in range(repeat):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = self.infer_features(X_dev, mask_dev, device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)
        avg_time, std_time = np.mean(times), np.std(times)
        print(f"Inference time: mean {avg_time:.3f} ms/call | std {std_time:.3f} ms")
        return avg_time


# ============================================================
# Main pipeline (single dataset; test_time supports multiple runs)
# ============================================================
def main(cfg, test_time=1):
    print("\n" + "#" * 96)
    print(f"# Dataset : {cfg.dataset_name} | slide_out: {cfg.slide_out}")
    print(f"# data_path : {cfg.data_path} | device: {device}")
    print(f"# CFG: { {k: v for k, v in vars(cfg).items()} }")
    print("#" * 96)

    # results_dir derived automatically (change the suffix on the next line to share a
    # directory with other versions)
    first_slide = str(cfg.slide_out).split(',')[0].strip()
    out_dir = os.path.join(f"./{cfg.dataset_name}_HGSF_results", first_slide)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'metrics.csv')
    print(f"# results_dir : {out_dir}")

    # ---------- 1. Data + preprocessing ----------
    X_tr, Y_tr_log, gene_names, coords_tr, _ = load_spa_data(cfg, mode='train')
    X_te, Y_te_log, _, coords_te, sid_te = load_spa_data(cfg, mode='test')

    if cfg.zscore_y:
        mu, sd = Y_tr_log.mean(0), Y_tr_log.std(0) + 1e-8
    else:
        mu = np.zeros(Y_tr_log.shape[1], dtype='float32')
        sd = np.ones(Y_tr_log.shape[1], dtype='float32')
    Y_tr = ((Y_tr_log - mu) / sd).astype('float32')
    Y_te = ((Y_te_log - mu) / sd).astype('float32')

    view_num = len(X_tr)
    N_tr, N_te = Y_tr.shape[0], Y_te.shape[0]
    print(f"Number of views: {view_num} | genes: {len(gene_names)}")
    if len(gene_names) < 200:
        print(f"warn: gene count {len(gene_names)} < 200, PCC-200 degrades to the top "
              f"{len(gene_names)} highly expressed genes")

    np.savez(os.path.join(out_dir, 'test_y.npz'),
             Y_z=Y_te, Y_log=Y_te_log, mu=mu, sd=sd,
             gene_names=np.array(gene_names))

    stats_runs = {'complete': [], 'missing': []}
    train_times, params_list = [], []

    for data_seed in range(1, test_time + 1):
        start_runtime = time.time()
        set_seed(cfg.seed + data_seed)

        # NEW: the reset also refreshes the new 'rad' / 'comp' keys
        loss_hist.clear()
        loss_hist.update({k: [] for k in ['rec', 'flow', 'vel', 'cluster',
                                          'entropy', 'rad', 'comp', 'total']})
        val_pcc_list.clear()
        run_sfx = f"_s{data_seed}" if test_time > 1 else ""

        # ---------- 2. Simulate missingness (missing-view features zeroed) ----------
        mask_tr_np = get_mask(view_num, N_tr, cfg.missing_rate,
                              seed=cfg.seed + data_seed)
        mask_tr_f = mask_tr_np.astype('float32')
        X_tr_masked = [X_tr[i] * mask_tr_f[:, i:i + 1] for i in range(view_num)]
        X_dev = [torch.from_numpy(x).float().to(device) for x in X_tr_masked]
        mask_dev = torch.from_numpy(mask_tr_np).long().to(device)

        complete = mask_tr_np.sum(1) == view_num
        print(f"[run {data_seed}] complete samples: {complete.sum()}/{N_tr}")

        rng = np.random.RandomState(cfg.seed + data_seed)
        perm = rng.permutation(N_tr)
        n_val = max(int(N_tr * cfg.val_ratio), 2)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        fit_idx = tr_idx[complete[tr_idx]] if cfg.fit_on == 'complete' else tr_idx

        # ---------- 3. Config (key names follow the original icdm) ----------
        ae_cfg = dict(batchnorm=True)
        for i in range(view_num):
            ae_cfg[f'arch{i + 1}'] = [int(X_tr[i].shape[1]),
                                      cfg.hidden, cfg.hidden, cfg.latent_dim]
            ae_cfg[f'activations{i + 1}'] = 'relu'

        config = dict(
            dataset=cfg.dataset_name,
            view=view_num,
            Autoencoder=ae_cfg,
            flow=dict(num_steps=cfg.flow_num_steps, beta_t=cfg.flow_beta_t,
                      eps=cfg.flow_eps, symmetric=cfg.flow_symmetric,
                      curv=cfg.flow_curv, ball_eps=cfg.flow_ball_eps,
                      auto_curv=cfg.flow_auto_curv,
                      auto_curv_target=cfg.flow_auto_curv_target,
                      auto_curv_epochs=cfg.flow_auto_curv_epochs),
            training=dict(n_clusters=cfg.n_clusters, dim=cfg.latent_dim,
                          n_epochs=cfg.train_epochs, eval_every=cfg.eval_freq,
                          batch_size=cfg.batch_size,
                          missing_rate=cfg.missing_rate,
                          loss_weights=dict(
                              vel_warmup_epoch=cfg.vel_warmup_epoch,
                              vel_dt_min=cfg.vel_dt_min,
                              vel_dt_max=cfg.vel_dt_max,
                              vel_bootstrap_missing=cfg.vel_bootstrap_missing,
                              cdc_num_steps=cfg.cdc_num_steps,  # NEW: L_cdc step=10
                          )),
        )

        # ---------- 4. Model + training ----------
        model = HGSF(config, cfg).to(device)
        params_M = sum(p.numel() for p in model.parameters()) / 1e6
        params_list.append(params_M)
        print(f"[run {data_seed}] model params: {params_M:.2f} M")

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        train_time, best_r = model.train_poinc(
            config, X_dev, mask_dev, Y_tr, tr_idx, val_idx, optimizer, device)
        train_times.append(train_time)

        plot_training_curves(out_dir, loss_hist, val_pcc_list, run_sfx)
        if cfg.run_diagnostics:
            model.count_flops_params(X_dev, mask_dev, config, device)
            model.measure_inference_time(X_dev, mask_dev, device)

        # ---------- 5. Test slides: fused features + Ridge + metrics + standalone plots ----------
        Z_tr = model.infer_features(X_dev, mask_dev, device)
        reg = make_pipeline(StandardScaler(), Ridge(alpha=cfg.reg_alpha))
        reg.fit(Z_tr[fit_idx], Y_tr[fit_idx])
        print(f"fused embedding: train {Z_tr.shape}")

        def run_eval(X_views_np, mask_np, base_tag):
            tag = base_tag + run_sfx
            X_d = [torch.from_numpy(x).float().to(device) for x in X_views_np]
            m_d = torch.from_numpy(mask_np).long().to(device)
            Z = model.infer_features(X_d, m_d, device)
            pred_log = reg.predict(Z) * sd + mu
            stats, pcc_g, pcc_s, rvd = compute_metrics(pred_log, Y_te_log)
            log_metrics(stats, f"{cfg.dataset_name} {tag}")

            plot_scatter_pred_true(pred_log, Y_te_log, out_dir, tag)
            plot_hist(pcc_g, out_dir, tag, "per-gene PCC", 'steelblue',
                      'fig_pcc_hist')
            plot_hist(pcc_s, out_dir, tag, "per-spot PCC (SpotPCC)",
                      'darkorange', 'fig_spotpcc_hist')
            plot_hist(np.clip(rvd, 0, 5), out_dir, tag,
                      "per-gene RVD (clipped at 5)", 'mediumpurple',
                      'fig_rvd_hist', extra_mean=False)
            plot_rvd_pcc(rvd, pcc_g, out_dir, tag)
            plot_top_genes(pcc_g, gene_names, out_dir, tag, k=cfg.topk_bar)

            if 'complete' in base_tag:
                if len(set(sid_te)) > 1:
                    sel = (sid_te == first_slide)
                    print(f"spatial maps on the first test slide: {first_slide} "
                          f"({int(sel.sum())}/{N_te} spots)")
                    plot_spatial_maps(coords_te[sel], Y_te_log[sel], pred_log[sel],
                                      gene_names, pcc_g, out_dir, tag,
                                      topk=cfg.spatial_topk)
                else:
                    plot_spatial_maps(coords_te, Y_te_log, pred_log, gene_names,
                                      pcc_g, out_dir, tag, topk=cfg.spatial_topk)

            pd.DataFrame(dict(
                gene=gene_names,
                PCC=pcc_g,
                RVD=rvd,
                MSE=np.mean((pred_log - Y_te_log) ** 2, 0),
                MAE=np.mean(np.abs(pred_log - Y_te_log), 0))
            ).to_csv(f"{out_dir}/per_gene_metrics_{tag}.csv", index=False)
            np.save(f"{out_dir}/pred_{tag}.npy", pred_log)
            save_metrics_csv(csv_path,
                             dict(dataset=cfg.dataset_name, MR=cfg.missing_rate,
                                  seed=data_seed, eval=base_tag), stats)
            return stats

        # ---- Scenario 1: complete views ----
        mask_te_full = np.ones((N_te, view_num), dtype='int64')
        stats_runs['complete'].append(
            run_eval(X_te, mask_te_full, f"MR{cfg.missing_rate}_complete"))

        # ---- Scenario 2: simulated missingness (prediction after flow imputation) ----
        if cfg.missing_rate > 0:
            mask_te = get_mask(view_num, N_te, cfg.missing_rate,
                               seed=cfg.seed + data_seed + 1000)
            mask_te_f = mask_te.astype('float32')
            X_te_masked = [X_te[i] * mask_te_f[:, i:i + 1]
                           for i in range(view_num)]
            stats_runs['missing'].append(
                run_eval(X_te_masked, mask_te, f"MR{cfg.missing_rate}_missing"))

        print(f"\n[run {data_seed}] Train: {train_time:.1f}s | "
              f"Total: {time.time() - start_runtime:.1f}s")

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 6. Aggregation ----------
    agg_c = aggregate_stats(stats_runs['complete'])
    agg_m = aggregate_stats(stats_runs['missing'])

    print("\n" + "=" * 90)
    print(f"Dataset: {cfg.dataset_name} | missing rate: {cfg.missing_rate} | "
          f"runs: {len(stats_runs['complete'])} | avg params: "
          f"{np.mean(params_list):.2f}M | avg train time: {np.mean(train_times):.1f}s")
    rows = []
    if agg_c is not None:
        print(f"complete test -> {fmt_metrics(agg_c)}")
        rows.append(dict(dataset=cfg.dataset_name, MR=cfg.missing_rate,
                         eval='complete', **agg_c))
    if agg_m is not None:
        print(f"missing test -> {fmt_metrics(agg_m)}")
        rows.append(dict(dataset=cfg.dataset_name, MR=cfg.missing_rate,
                         eval='missing', **agg_m))
    print("=" * 90 + "\n")

    comp_stats = {t: s for t, s in [('complete', agg_c), ('missing', agg_m)]
                  if s is not None}
    plot_metric_comparison(comp_stats, out_dir)

    with open(f"{out_dir}/summary_metrics.json", 'w') as f:
        json.dump(dict(per_run=stats_runs, mean=comp_stats), f, indent=2)
    print(f"All plots and results saved to: {out_dir}")
    return rows


def run_all(dataset_cfgs=None, test_time=1):
    if dataset_cfgs is None:
        dataset_cfgs = DATASETS
    grand = []
    for ds in dataset_cfgs:
        grand += main(make_cfg(ds), test_time=test_time)
    print_final_summary(grand)


if __name__ == '__main__':
    test_time = 1
    # All four datasets; to run a single one: dataset_cfgs=[DATASETS[3]] (her2st)
    run_all(dataset_cfgs=DATASETS, test_time=test_time)
