# -*- coding: utf-8 -*-
"""
config.py -- Unified configuration: dataset registry + global hyperparameters
             (+ legacy multi-view clustering configs).
All data paths are anonymized: the unified data root is /data.
"""
import numpy as np

# ============================================================
# Dataset registry: slide naming = prefix + number
#   her2st -> SPA + number  (SPA119.pt)
#   PRAD   -> MEND + number (MEND139.pt)
#   cSCC   -> NCBI + number (NCBI759.pt)
#   DLPFC  -> MISC + number (MISC1.pt)
# ============================================================
DATA_ROOT = "/data/hest1k_datasets"

DATASETS = [
    dict(expr_name="DLPFC",  prefix="MISC", slide_out="MISC1",
         data_path=f"{DATA_ROOT}/DLPFC/"),
    dict(expr_name="PRAD",   prefix="MEND", slide_out="MEND139",
         data_path=f"{DATA_ROOT}/PRAD/"),
    dict(expr_name="cSCC",   prefix="NCBI", slide_out="NCBI759",
         data_path=f"{DATA_ROOT}/cSCC/"),
    dict(expr_name="her2st", prefix="SPA",  slide_out="SPA119",  
         data_path=f"{DATA_ROOT}/her2st/"),
]


# ============================================================
# Global configuration (data_path / slide_out are overridden by DATASETS)
# ============================================================
class CFG:
    # ---- Data (overridden by DATASETS) ----
    dataset_name = 'her2st'
    data_path = f'{DATA_ROOT}/her2st/'
    slidename_list = 'all_slide_lst.txt'   # shared with the processed_data of other versions
    gene_list = 'selected_gene_list.txt'
    slide_out = 'SPA119'

    # ---- View processing ----
    neighbor_mode = 'mean'
    global_mode = 'mean'
    log_y = True
    zscore_y = True
    val_ratio = 0.1

    # ---- Missing-value simulation ----
    missing_rate = 0.5
    fit_on = 'complete'

    # ---- Network ----
    latent_dim = 128
    hidden = 1024
    n_clusters = 128

    # ---- Training ----
    lr = 1e-4
    batch_size = 256
    train_epochs = 200
    eval_freq = 1
    seed = 0

    # ---- Flow ----
    flow_num_steps = 1
    flow_beta_t = 0.5
    flow_eps = 1e-4
    flow_symmetric = True       # symmetric augmentation (random ordered view pairs)
    flow_curv = 1e-9            # Euclidean warmup, paired with auto_curv
    flow_ball_eps = 1e-4
    flow_auto_curv = True
    flow_auto_curv_target = 0.9
    flow_auto_curv_epochs = [3, 30]

    # ---- CDC (former velocity-consistency weight slot) ----
    vel_warmup_epoch = 20
    vel_dt_min = 0.3            # no longer used after the CDC replacement; kept for compatibility
    vel_dt_max = 0.5            # ditto
    vel_bootstrap_missing = True
    cdc_num_steps = 10          # NEW: L_cdc step budget N (h = 1/N), step = 10

    # ---- Regressor ----
    reg_alpha = 10.0

    # ---- Loss weights [L1(flow), L2(cdc), L3(cluster), L4(entropy),
    #      L5(rad=1e-4), L6(comp=1e-4)] ----
    lamda_List = [[1.0, 1.0, 1.0, 1.0, 0.0001, 0.0001]]
    flagwithAblation = 0

    # ---- Diagnostics ----
    run_diagnostics = False     # True: run FLOPs / inference-time profiling after training (needs thop)

    # ---- Output ----
    gpu = '0'
    spatial_topk = 5
    topk_bar = 20


def make_cfg(ds=None):
    cfg = CFG()
    for k, v in vars(CFG).items():
        if not k.startswith('_'):
            setattr(cfg, k, v)
    if ds is not None:
        cfg.dataset_name = ds['expr_name']
        cfg.data_path = ds['data_path']
        cfg.slide_out = ds['slide_out']
    return cfg


# ============================================================
# legacy: default configs for multi-view clustering datasets
# ============================================================
def get_default_config(data_name):
    if data_name in ['CUB']:
        return dict(
            Autoencoder=dict(
                arch1=[1024, 512, 1024, 1024, 128],
                arch2=[300, 512, 1024, 1024, 128],
                activations1='sigmoid',
                activations2='relu',
                batchnorm=True,
            ),
         
        )
    elif data_name in ['LandUse_21']:
        return dict(
          
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 128],
                arch2=[40, 1024, 1024, 1024, 128],
                activations1='relu', activations2='relu', batchnorm=True,
            ),
    elif data_name in ['handwrite']:
        return dict(
            Autoencoder=dict(
                arch1=[76, 512, 1024, 1024, 128],
                arch2=[64, 512, 1024, 1024, 128],
                activations1='sigmoid', activations2='relu', batchnorm=True,
            ),
        )
    elif data_name in ['Synthetic3d']:
        return dict(
            Autoencoder=dict(
                arch1=[3, 1024, 1024, 1024, 128],
                arch2=[3, 1024, 1024, 1024, 128],
                activations1='relu', activations2='relu', batchnorm=True,
            ),
        )
    elif data_name in ['Sence_15']:
        return dict(
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 128],
                arch2=[40, 1024, 1024, 1024, 128],
                activations1='relu', activations2='relu', batchnorm=True,
            ),
        )
    elif data_name in ['Multi-Fashion']:
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[784, 1024, 1024, 1024, 128],
                activations1='relu', activations2='relu', batchnorm=True,
            ),
        )
    else:
        raise Exception('Undefined data_name')
