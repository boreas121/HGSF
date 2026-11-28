# -*- coding: utf-8 -*-
"""
visualization.py -- Plotting (one PNG per figure).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_scatter_pred_true(pred, true, out_dir, tag, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    n = min(100000, pred.size)
    idx = rng.choice(pred.size, n, replace=False)
    x, y = true.flatten()[idx], pred.flatten()[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(x, y, gridsize=80, bins='log', cmap='viridis', mincnt=1)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_title(f"{tag} | global PCC={np.corrcoef(x, y)[0, 1]:.4f}")
    ax.set_xlabel("true (log2)")
    ax.set_ylabel("predicted (log2)")
    fig.colorbar(hb, ax=ax, label='count(log)')
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_scatter_{tag}.png", dpi=150)
    plt.close(fig)


def plot_hist(values, out_dir, tag, xlabel, color, fname, extra_mean=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    v = values[~np.isnan(values)]
    ax.hist(v, bins=50, color=color, alpha=0.8)
    ax.axvline(v.mean(), color='r', ls='--', label=f"mean={v.mean():.3f}")
    if extra_mean:
        ax.axvline(np.median(v), color='g', ls='--', label=f"median={np.median(v):.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{fname}_{tag}.png", dpi=150)
    plt.close(fig)


def plot_top_genes(pcc_g, gene_names, out_dir, tag, k=20):
    v = np.nan_to_num(pcc_g, nan=-1)
    order = np.argsort(-v)[:k][::-1]
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh([gene_names[i] for i in order], [pcc_g[i] for i in order], color='teal')
    ax.set_xlabel("PCC")
    ax.set_title(f"{tag} top-{k} genes by PCC")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_top_genes_{tag}.png", dpi=150)
    plt.close(fig)


def plot_spatial_maps(coords, true, pred, gene_names, pcc_g, out_dir, tag, topk=5):
    v = np.nan_to_num(pcc_g, nan=-1)
    for g in np.argsort(-v)[:topk]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, mat, name in zip(axes, [true, pred], ["true", "predicted"]):
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=mat[:, g], cmap='magma', s=8)
            ax.set_title(f"{gene_names[g]} ({name}) | PCC={pcc_g[g]:.3f}")
            ax.set_aspect('equal')
            ax.axis('off')
        fig.colorbar(sc, ax=ax, shrink=0.8, label='log2 expr')
        fig.suptitle(f"{tag} spatial map")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/fig_spatial_{gene_names[g]}_{tag}.png", dpi=150)
        plt.close(fig)


def plot_rvd_pcc(rvd, pcc_g, out_dir, tag):
    fig, ax = plt.subplots(figsize=(6, 5))
    ok = ~np.isnan(pcc_g)
    ax.scatter(pcc_g[ok], np.clip(rvd[ok], 0, 5), s=8, alpha=0.6, color='purple')
    ax.set_xlabel("per-gene PCC")
    ax.set_ylabel("RVD (clipped at 5)")
    ax.set_title(f"{tag} RVD vs PCC")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_rvd_vs_pcc_{tag}.png", dpi=150)
    plt.close(fig)


def plot_training_curves(out_dir, loss_hist, val_pcc_list, sfx=''):
    x = range(len(loss_hist['total']))
    # NEW: the 'rad' and 'comp' training curves
    for k in ['rec', 'flow', 'vel', 'cluster', 'entropy', 'rad', 'comp', 'total']:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, loss_hist[k], color='steelblue')
        ax.grid()
        ax.set_xlabel("epoch")
        ax.set_ylabel(k)
        if k == 'cluster':
            title = "loss: cluster (cross-view NT-Xent)"
        elif k == 'vel':
            title = "loss: vel (CDC, cumulative displacement consistency)"
        elif k == 'rad':
            title = "loss: rad (radius consistency, L_rad)"
        elif k == 'comp':
            title = "loss: comp (cross-completed NT-Xent, L_comp)"
        else:
            title = f"loss: {k}"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/fig_train_loss_{k}{sfx}.png", dpi=150)
        plt.close(fig)
    if val_pcc_list:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(len(val_pcc_list)), val_pcc_list, color='crimson')
        ax.grid()
        ax.set_xlabel("eval #")
        ax.set_ylabel("val gene-PCC")
        ax.set_title("validation gene-PCC during training")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/fig_val_pcc{sfx}.png", dpi=150)
        plt.close(fig)


def plot_metric_comparison(all_stats, out_dir):
    metric_names = ['PCC_mean', 'PCC-10', 'PCC-50', 'PCC-200',
                    'MAE', 'MSE', 'RMSE', 'SpotPCC', 'SpotPCC_median',
                    'RVD', 'RVD_median']
    lower_better = ('MSE', 'MAE', 'RMSE', 'RVD', 'RVD_median')
    tags = list(all_stats.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tags), 2)))
    for m in metric_names:
        if not all(m in s for s in all_stats.values()):
            continue
        fig, ax = plt.subplots(figsize=(4.5, 4))
        vals = [all_stats[t][m] for t in tags]
        bars = ax.bar(tags, vals, color=colors[:len(tags)], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylabel(m)
        ax.set_title(m + (" (lower is better)" if m in lower_better
                          else " (higher is better)"))
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/fig_compare_{m}.png", dpi=150)
        plt.close(fig)
