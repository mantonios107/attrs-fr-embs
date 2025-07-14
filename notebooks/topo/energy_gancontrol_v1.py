import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import os
import pickle
from scipy import stats
import random
import time


embeddings_path = "../../data/embeddings/embeddings/"
models = ["facenet","arcface"]
attributes = ['0','1','2','3','4','brightness', 'contrast', 'hue']
max_curves = 5*10**3 # Number of curves to be sampled

energy = {m:[] for m in models}

# Precompute necessary quantities
metadata = pd.read_csv(os.path.join(embeddings_path,"df_param_step.csv"))
metadata = metadata.assign(moving_dimension=lambda x: x['moving_dimension'].astype("str")) # Fix problem in metadata
metadatas = {dim: metadata.loc[metadata["moving_dimension"]==dim] for dim in attributes} # Filter the metadas for each attribute
ids_curves = {dim:np.unique(metadatas[dim]["id_curve"].values) for dim in attributes}
for dim in ids_curves.keys():
    random.shuffle(ids_curves[dim])

# Sample the curves
adfs = {dim: metadatas[dim][metadatas[dim]["id_curve"].isin(ids_curves[dim][:max_curves])].sort_values("id_curve") for dim in attributes}

# Compute the indices of the emebeddings
adfs_0 = {dim: adfs[dim][adfs[dim]["param_step"]==0]["index"].values for dim in attributes}
adfs_1 = {dim: adfs[dim][adfs[dim]["param_step"]==1]["index"].values for dim in attributes}
adfs_2 = {dim: adfs[dim][adfs[dim]["param_step"]==2]["index"].values for dim in attributes}

for m in models:
    t_start = time.time()
    print("MODEL: ", m)
    path = os.path.join(embeddings_path, m)
    # for p in tqdm(os.listdir(embeddings_path + m + "/")):  # iterate over identities
    # if p.split(".")[1] == "npy":
    for p in tqdm([f for f in os.listdir(path) if f.endswith(".npy")]):

        # Load embeddings
        embeddings = np.load(os.path.join(embeddings_path + m + "/", p))
        en = []
        for dim in attributes:

            embs_pc = embeddings[adfs_1[dim], :]  # point cloud
            vf_pc = embeddings[adfs_2[dim], :] - embeddings[adfs_0[dim], :]  # vector field (3 point stencil)
            vf_pc = (vf_pc.T / np.linalg.norm(vf_pc, axis=1)).T  # normalize vector field

            # Compute distance matrix
            if m == "facenet":
                D_mat = euclidean_distances(embs_pc)
            else:
                D_mat = cosine_distances(embs_pc)

            D_mat[D_mat == 0] = np.inf
            inv_D_mat = np.nan_to_num(1 / D_mat, posinf=0)
            D_mat_vf = cosine_distances(vf_pc)  # Compute cosine distances between point clouds

            en.append(np.mean(np.sum(inv_D_mat * D_mat_vf, 1) / np.sum(inv_D_mat, 1)))

        energy[m].append(en)
    print(f"Time taken for model {m}: {time.time() - t_start:.2f} seconds")

# MODEL:  facenet
# 100%|██████████| 3/3 [00:13<00:00,  4.60s/it]
#   0%|          | 0/3 [00:00<?, ?it/s]MODEL:  arcface
# 100%|██████████| 3/3 [00:09<00:00,  3.05s/it]