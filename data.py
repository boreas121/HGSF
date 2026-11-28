# -*- coding: utf-8 -*-
"""
data.py -- Data loading.
load_spa_data: SPA three views (local / neighbor / global) + gene expression (main pipeline)
load_data:     legacy multi-view .mat data (paths anonymized to /data)
"""
import sys

import numpy as np
import pandas as pd
import anndata
import scipy.io as sio

from utils import read_txt_list, torch_load_safe, to_np


# ============================================================
# SPA three-view data (main pipeline)
# ============================================================
def load_spa_data(cfg, mode='train'):
    data_path, slide_out = cfg.data_path, cfg.slide_out
    slide_list = read_txt_list(data_path + 'processed_data/' + cfg.slidename_list)
    slide_out_list = [s.strip() for s in str(slide_out).split(',') if s.strip()]
    if mode == 'train':
        for so in slide_out_list:
            if so in slide_list:
                slide_list.remove(so)
                print(f"{so} is removed for testing")
            else:
                print(f"warn: {so} is not in slidename_list")
    else:
        slide_list = slide_out_list

    gene_names = read_txt_list(data_path + 'processed_data/' + cfg.gene_list)
    print(f'len selected genes : {len(gene_names)}')

    all_gene, all_local, all_nb, all_gb, all_xy, all_sid = [], [], [], [], [], []
    for slide in slide_list:
        a = anndata.read_h5ad(data_path + 'st/' + slide + '.h5ad')
        Xsub = a[:, gene_names].X
        gene_mtx = Xsub.toarray() if hasattr(Xsub, 'toarray') else np.asarray(Xsub)
        gene_mtx = gene_mtx.astype('float32')

        local_ebd = to_np(torch_load_safe(data_path + 'processed_data/local_ebd/' + slide + '.pt'))
        nb_ebd = to_np(torch_load_safe(data_path + 'processed_data/neighbor_ebd/' + slide + '.pt'))
        gb_ebd = to_np(torch_load_safe(data_path + 'processed_data/global_ebd/' + slide + '.pt'))
        xy = np.asarray(a.obsm['spatial'][:, :2], dtype='float32')
        n = gene_mtx.shape[0]
        assert local_ebd.shape[0] == nb_ebd.shape[0] == gb_ebd.shape[0] == n, \
            f"{slide}: spot count mismatch gene={n}, local={local_ebd.shape[0]}, " \
            f"neighbor={nb_ebd.shape[0]}, global={gb_ebd.shape[0]}"

        all_gene.append(gene_mtx)
        all_local.append(local_ebd)
        all_nb.append(nb_ebd)
        all_gb.append(gb_ebd)
        all_xy.append(xy)
        all_sid.extend([slide] * n)
        print(f"{slide} loaded, gene mtx: {gene_mtx.shape}, local: {local_ebd.shape}, "
              f"neighbor: {nb_ebd.shape}, global: {gb_ebd.shape}")

    gene_mtx, local = np.concatenate(all_gene, 0), np.concatenate(all_local, 0)
    nb, gb = np.concatenate(all_nb, 0), np.concatenate(all_gb, 0)
    coords, slide_ids = np.concatenate(all_xy, 0), np.asarray(all_sid)

    gene_df = pd.DataFrame(gene_mtx)
    all_nan_idx = gene_df.index[gene_df.isnull().all(axis=1)]
    all_zero_idx = gene_df.index[gene_df.sum(axis=1) == 0]
    print(f"All NAN spot index: {all_nan_idx}")
    print(f"All zero spot index: {all_zero_idx}")
    keep = sorted(set(gene_df.index) - (set(all_nan_idx) | set(all_zero_idx)))
    gene_mtx, local = gene_mtx[keep], local[keep]
    nb, gb = nb[keep], gb[keep]
    coords, slide_ids = coords[keep], slide_ids[keep]

    Y = (np.log2(gene_mtx + 1.0) if cfg.log_y else gene_mtx.copy()).astype('float32')
    x_nb = nb.mean(axis=1) if cfg.neighbor_mode == 'mean' else nb.reshape(nb.shape[0], -1)
    x_gb = gb.mean(axis=1) if cfg.global_mode == 'mean' else gb.reshape(gb.shape[0], -1)
    X_list = [local.astype('float32'), x_nb.astype('float32'), x_gb.astype('float32')]
    print(f"{cfg.dataset_name}[{mode}] ready: view0={X_list[0].shape}, "
          f"view1={X_list[1].shape}, view2={X_list[2].shape}, Y={Y.shape}")
    return X_list, Y, gene_names, coords, slide_ids


# ============================================================
# legacy: multi-view .mat data (paths anonymized: /data)
# ============================================================
def load_data(config):
    """Load data"""
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    print("shuffle")
    if data_name in ['CUB']:
        mat = sio.loadmat("/data/cub_googlenet_doc2vec_c10.mat")
        X_list.append(mat['X'][0][0].astype('float32'))
        X_list.append(mat['X'][0][1].astype('float32'))
        Y_list.append(np.squeeze(mat['gt']))
    elif data_name in ['LandUse_21']:
        mat = sio.loadmat("/data/LandUse_21.mat")
        X_list.append(mat['X'][0][1].astype('float32'))
        X_list.append(mat['X'][0][2].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['Sence_15']:
        mat = sio.loadmat("/data/Scene15.mat")
        X_list.append(mat['X'][0][1].astype('float32'))
        X_list.append(mat['X'][0][2].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['handwrite']:
        mat = sio.loadmat("/data/Handwritten.mat")
        X_list.append(mat['X'][1][0].astype('float32'))
        X_list.append(mat['X'][2][0].astype('float32'))
        Y_list.append(np.squeeze(mat['y']))
    elif data_name in ['Multi-Fashion']:
        mat = sio.loadmat("/data/Fashion.mat")
        X_list.append(mat['X1'].reshape(-1, 784).astype('float32'))
        X_list.append(mat['X2'].reshape(-1, 784).astype('float32'))
        Y_list.append(np.squeeze(mat['Y'].T))
    elif data_name in ['Synthetic3d']:
        mat = sio.loadmat("/data/synthetic3d.mat")
        X_list.append(mat['X'][1][0].astype('float32'))
        X_list.append(mat['X'][2][0].astype('float32'))
        Y_list.append(np.squeeze(mat['Y'].T))
    return X_list, Y_list
