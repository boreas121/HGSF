# -*- coding: utf-8 -*-
"""
preprocess.py -- H&E patch-embedding extraction and gene list selection (from preprocess.ipynb)
1) Read the per-slide ST h5ad files and compute the common gene set
2) Extract per-spot patch embeddings with UNI (MahmoodLab, ViT-L/16):
   local [N,1536] / neighbor [N,9,1536] / global [N,49,1536]
3) Union HVG -> intersect the top-mean and top-std genes -> final gene list
Paths anonymized: /data/hest1k_datasets/
"""
import os
from tqdm import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pandas as pd
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import scanpy as sc
import anndata
from scipy.spatial import KDTree
import torch

from huggingface_hub import login

# ========== Paths (anonymized; the raw folder was named Her2st/, same data as the training-side her2st) ==========
data_path = "/data/hest1k_datasets/her2st/"
tif_path = data_path + 'wsis/'
st_path = data_path + 'st/'

# ========== 1. Load UNI ==========
login("")  # login with your User Access Token, found at https://huggingface.co/settings/tokens

model_UNI = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                              init_values=1e-5, dynamic_img_size=True)
transform_UNI = create_transform(**resolve_data_config(model_UNI.pretrained_cfg, model=model_UNI))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_UNI.eval()
model_UNI.to(device)


def get_img_ebd_uni(patch, model=model_UNI, transform=transform_UNI):
    base_width = 224
    patch_resized = patch.resize((base_width, base_width), Image.Resampling.LANCZOS)  # [224, 224]
    img_transformed = transform(patch_resized).unsqueeze(0).to(device)
    with torch.inference_mode():
        feature_emb = model(img_transformed)
    return torch.clone(feature_emb)


# ========== 2. Read ST h5ad and compute the common gene set ==========
st_data_list = []
file_name_list = ["SPA" + str(i) for i in range(119, 155)]

first = True
for fn in file_name_list:
    st_data = anndata.read_h5ad(st_path + fn + '.h5ad')
    st_data_list.append(st_data)
    if first:
        commen_genes = st_data.var_names
        first = False
        print(fn, st_data.shape)
        continue
    commen_genes = set(commen_genes).intersection(set(st_data.var_names))
    print(fn, st_data.shape, end='\t')

print("length of commen genes: ", len(commen_genes))
commen_genes = sorted(list(commen_genes))
for fni in range(len(file_name_list)):
    st_data = st_data_list[fni].copy()
    st_data_list[fni] = st_data[:, commen_genes].copy()
    print(file_name_list[fni], " ", st_data_list[fni].shape)
print('Only keep commen genes')

# ========== 3. Output directories ==========
os.makedirs(data_path + 'processed_data/', exist_ok=True)
os.makedirs(data_path + 'processed_data/local_ebd/', exist_ok=True)
os.makedirs(data_path + 'processed_data/neighbor_ebd/', exist_ok=True)
os.makedirs(data_path + 'processed_data/global_ebd/', exist_ok=True)
save_path = data_path + 'processed_data/'

# ========== 4. Extract the three-view embeddings and save ==========
for i in range(len(file_name_list)):
    fn = file_name_list[i]
    st_data = st_data_list[i]
    print(fn)
    img = Image.open(tif_path + fn + ".tif")
    x = st_data.obsm["spatial"][:, 0]  # x coordinate in the H&E image
    y = st_data.obsm["spatial"][:, 1]  # y coordinate in the H&E image
    coords = np.stack([x, y], axis=1)
    tree = KDTree(coords)
    spot_diameter = st_data.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
    print(f" {fn} | Spot diameter: ", spot_diameter)  # Spot diameter for Visium
    if spot_diameter < 224:
        radius = 112  # minimum patch size: 224 by 224
    else:
        radius = int(spot_diameter // 2)
    all_local_ebd = None
    global_ebd = []
    neighbor_ebd = []
    all_neighbor_ebd = []
    all_global_ebd = []
    first = True
    for spot_idx in tqdm(range(len(x))):
        patch = img.crop((x[spot_idx] - radius, y[spot_idx] - radius,
                          x[spot_idx] + radius, y[spot_idx] + radius))
        patch_ebd_uni = get_img_ebd_uni(patch)
        # patch_ebd_conch = get_img_embd_conch(patch)
        # patch_ebd = torch.cat([patch_ebd_uni, patch_ebd_conch], dim=1)
        if first:
            all_local_ebd = patch_ebd_uni
            first = False
            continue
        all_local_ebd = torch.cat((all_local_ebd, patch_ebd_uni), dim=0)
        for j in range(len(coords)):
            _, n_idx = tree.query(coords[j], k=9)
            _, g_idx = tree.query(coords[j], k=49)
            neighbor_ebd.append(all_local_ebd[n_idx].unsqueeze(0))  # [1, 9, D]
            global_ebd.append(all_local_ebd[g_idx].unsqueeze(0))    # [1, 49, D]

    all_neighbor_ebd = torch.cat(neighbor_ebd, dim=0)  # [N_i, 9, D]
    all_global_ebd = torch.cat(global_ebd, dim=0)      # [N_i, 49, D]
    print(f' local ebd shape: {all_local_ebd.shape} | neighbor ebd shape: '
          f'{all_neighbor_ebd.shape} | global ebd shape: {all_global_ebd.shape}')
    if save_path is not None:
        torch.save(all_neighbor_ebd.detach().cpu(), save_path + 'neighbor_ebd/' + fn + ".pt")
        torch.save(all_local_ebd.detach().cpu(), save_path + 'local_ebd/' + fn + ".pt")
        torch.save(all_global_ebd.detach().cpu(), save_path + 'global_ebd/' + fn + ".pt")
    print("#" * 20)

# ========== 5. Union HVG ==========
union_hvg = set()

for fn_idx in range(len(file_name_list)):
    st_data = st_data_list[fn_idx].copy()
    fn = file_name_list[fn_idx]

    sc.pp.filter_cells(st_data, min_genes=1)
    sc.pp.filter_genes(st_data, min_cells=1)
    sc.pp.normalize_total(st_data, inplace=True)
    sc.pp.log1p(st_data)
    sc.pp.highly_variable_genes(st_data, n_top_genes=2000)

    union_hvg = union_hvg.union(set(st_data.var_names[st_data.var["highly_variable"]]))
    print(fn, len(union_hvg))

union_hvg = sorted([gene for gene in union_hvg])
print(len(union_hvg))

# ========== 6. Select union_hvg and concatenate all slides ==========
all_count_df = pd.DataFrame(st_data_list[0][:, union_hvg].X.toarray(),
                            columns=union_hvg,
                            index=[file_name_list[0] + "_" + str(i)
                                   for i in range(st_data_list[0].shape[0])]).T

for fn_idx in range(0, len(file_name_list)):
    st_data = st_data_list[fn_idx]
    df = pd.DataFrame(st_data[:, union_hvg].X.toarray(),
                      columns=union_hvg,
                      index=[file_name_list[fn_idx] + "_" + str(i)
                             for i in range(st_data.shape[0])]).T
    all_count_df = pd.concat([all_count_df, df], axis=1)
    print(file_name_list[fn_idx], st_data.shape, all_count_df.shape)

all_count_df.fillna(0, inplace=True)
all_count_df = all_count_df.T

# ========== 7. Order the selected genes by mean and std ==========
all_gene_order_by_mean = all_count_df.mean(axis=0).sort_values(ascending=False).index
all_gene_order_by_std = all_count_df.std(axis=0).sort_values(ascending=False).index

# ========== 8. Take the intersection of high-mean and high-variance genes ==========
num_genes = 300  # to make the final gene list of length 200

selected_genes = sorted(list(set(all_gene_order_by_mean[:num_genes]).intersection(
    set(all_gene_order_by_std[:num_genes]))))
print(len(selected_genes))

# ========== 9. Save the gene list and the slide list ==========
with open(data_path + "processed_data/selected_gene_list.txt", "w") as f:
    for gene in selected_genes:
        f.write(gene + "\n")

with open(data_path + "processed_data/all_slide_lst.txt", "w") as f:
    for fn in file_name_list:
        f.write(fn + "\n")
