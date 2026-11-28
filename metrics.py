# -*- coding: utf-8 -*-
"""
metrics.py
"""
import os

import numpy as np
import pandas as pd


# ============================================================
# Per-gene / per-spot PCC
# ============================================================
def per_gene_pearson(pred, true):
    p = pred - pred.mean(0, keepdims=True)
    t = true - true.mean(0, keepdims=True)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    r = (p * t).sum(0) / np.maximum(den, 1e-12)
    r[(t ** 2).sum(0) == 0] = np.nan
    return r


def per_spot_pearson(pred, true):
    p = pred - pred.mean(1, keepdims=True)
    t = true - true.mean(1, keepdims=True)
    den = np.sqrt((p ** 2).sum(1) * (t ** 2).sum(1))
    r = (p * t).sum(1) / np.maximum(den, 1e-12)
    r[(t ** 2).sum(1) == 0] = np.nan
    return r


def compute_metrics(pred, true, topk_list=(10, 50, 200)):
    """Conventions identical to the previous versions: PCC-N = mean per-gene PCC
    over the top-N most highly expressed genes; MAE/MSE in log2 space;
    RVD = per-gene std relative deviation"""
    err = pred - true
    pcc_g = per_gene_pearson(pred, true)
    pcc_s = per_spot_pearson(pred, true)
    std_t, std_p = true.std(0), pred.std(0)
    valid = std_t > 1e-6
    rvd = np.full(true.shape[1], np.nan, dtype='float64')
    rvd[valid] = np.abs(std_p[valid] - std_t[valid]) / std_t[valid]
    G = true.shape[1]
    order = np.argsort(-true.mean(0))
    stats = dict(
        MSE=float(np.mean(err ** 2)),
        MAE=float(np.mean(np.abs(err))),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        PCC_mean=float(np.nanmean(pcc_g)),
        PCC_median=float(np.nanmedian(pcc_g)),
        SpotPCC=float(np.nanmean(pcc_s)),
        SpotPCC_median=float(np.nanmedian(pcc_s)),
        RVD=float(np.nanmean(rvd)),
        RVD_median=float(np.nanmedian(rvd)),
        n_valid_genes=int((~np.isnan(pcc_g)).sum()),
    )
    for k in topk_list:
        kk = min(int(k), G)
        stats[f'PCC-{k}'] = float(np.nanmean(pcc_g[order[:kk]]))
    return stats, pcc_g, pcc_s, rvd


def log_metrics(stats, tag):
    print(f"[{tag}] " + " | ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in stats.items()))


# ============================================================
# Aggregation utilities (same format as the previous versions)
# ============================================================
def aggregate_stats(stats_list):
    if not stats_list:
        return None
    return {k: float(np.nanmean([float(s[k]) for s in stats_list]))
            for k in stats_list[0]}


def fmt_metrics(d):
    keys = ['PCC_mean', 'PCC-10', 'PCC-50', 'PCC-200', 'MAE', 'MSE', 'RVD']
    return " | ".join(f"{k}={d[k]:.4f}" for k in keys if k in d)


def save_metrics_csv(csv_path, meta, stats):
    try:
        row = {**meta, **stats}
        df = pd.DataFrame([row])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"warn: failed to write metrics.csv: {e}")


def print_final_summary(rows, save_csv='./HGSF_all_datasets_summary.csv'):
    if not rows:
        return
    line_len = 10 + 8 + 10 + 12 * 7
    print("\n" + "=" * line_len)
    print(f"{'dataset':<10}{'MR':<8}{'eval':<10}"
          f"{'PCC':<12}{'PCC-10':<12}{'PCC-50':<12}{'PCC-200':<12}"
          f"{'MAE':<12}{'MSE':<12}{'RVD':<12}")
    print("-" * line_len)
    for row in rows:
        def _c(name):
            v = row.get(name, float('nan'))
            try:
                return f"{v:<12.4f}"
            except (ValueError, TypeError):
                return f"{str(v):<12.12}"
        print(f"{str(row.get('dataset', '')):<10}{str(row.get('MR', '')):<8}"
              f"{str(row.get('eval', '')):<10}"
              + _c('PCC_mean') + _c('PCC-10') + _c('PCC-50') + _c('PCC-200')
              + _c('MAE') + _c('MSE') + _c('RVD'))
    print("=" * line_len)
    try:
        pd.DataFrame(rows).to_csv(save_csv, index=False)
        print(f"Summary saved: {save_csv}")
    except Exception as e:
        print(f"warn: failed to save the summary: {e}")
