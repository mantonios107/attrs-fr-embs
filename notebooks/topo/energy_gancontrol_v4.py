import os
import random
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial


print("JAX is using device:", jax.devices())

embeddings_path = "../../data/embeddings/embeddings/"
models = ["facenet","arcface"]
attributes = ['0','1','2','3','4','brightness', 'contrast', 'hue']
max_curves = 5*10**3

# Load and preprocess metadata
metadata = pd.read_csv(os.path.join(embeddings_path,"df_param_step.csv"))
metadata = metadata.assign(moving_dimension=lambda x: x['moving_dimension'].astype("str"))

metadatas = {dim: metadata.loc[metadata["moving_dimension"]==dim] for dim in attributes}
ids_curves = {dim: np.unique(metadatas[dim]["id_curve"].values) for dim in attributes}
for dim in ids_curves.keys():
    random.shuffle(ids_curves[dim])

adfs = {
    dim: metadatas[dim][metadatas[dim]["id_curve"].isin(ids_curves[dim][:max_curves])].sort_values("id_curve")
    for dim in attributes
}

# Compute indices for each attribute and step
adfs_0 = {dim: adfs[dim][adfs[dim]["param_step"] == 0]["index"].values for dim in attributes}
adfs_1 = {dim: adfs[dim][adfs[dim]["param_step"] == 1]["index"].values for dim in attributes}
adfs_2 = {dim: adfs[dim][adfs[dim]["param_step"] == 2]["index"].values for dim in attributes}

# Convert to jnp arrays and stack along a new dimension for attributes
# Assuming each attribute set has the same number of curves after sampling (max_curves)
adfs_0_all = jnp.stack([jnp.array(adfs_0[dim]) for dim in attributes])  # shape: (num_attributes, C)
adfs_1_all = jnp.stack([jnp.array(adfs_1[dim]) for dim in attributes])  # shape: (num_attributes, C)
adfs_2_all = jnp.stack([jnp.array(adfs_2[dim]) for dim in attributes])  # shape: (num_attributes, C)

def euclidean_distances_jax(X):
    # X shape: (C, D)
    norms = jnp.sum(X**2, axis=1, keepdims=True)
    D = jnp.sqrt(jnp.maximum(norms + norms.T - 2 * (X @ X.T), 0))
    return D

def cosine_distances_jax(X):
    # X shape: (C, D)
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    sim = (X / norms) @ (X / norms).T
    return 1.0 - sim



@partial(jax.jit, static_argnames=['model_type'])
def compute_energy_for_identity(embeddings, model_type):
    # embeddings: (M, D) for all embeddings of the identity
    # We will index embeddings with (num_attributes, C) arrays.

    # Extract the point clouds (embs_pc) and vector fields (vf_pc) for all attributes at once.
    # embs_pc shape: (num_attributes, C, D)
    embs_pc = embeddings[adfs_1_all, :]
    vf_pc = embeddings[adfs_2_all, :] - embeddings[adfs_0_all, :]

    # Normalize vf_pc along the last dimension
    vf_norms = jnp.linalg.norm(vf_pc, axis=2, keepdims=True)
    vf_pc = vf_pc / vf_norms

    # Compute distance matrices for all attributes
    if model_type == "facenet":
        D_mat = euclidean_distances_batched(embs_pc)
    else:
        D_mat = cosine_distances_batched(embs_pc)

    # Replace zero distances with inf
    D_mat = jnp.where(D_mat == 0, jnp.inf, D_mat)
    inv_D_mat = 1.0 / D_mat
    inv_D_mat = jnp.nan_to_num(inv_D_mat, posinf=0.0)

    if model_type == "facenet":
        # Distance matrix for vf_pc
        D_mat_vf = euclidean_distances_batched(vf_pc)
    else:
        # Distance matrix for vf_pc
        D_mat_vf = cosine_distances_batched(vf_pc)  # (num_attributes, C, C)

    # Compute energies
    num = jnp.sum(inv_D_mat * D_mat_vf, axis=2)  # (num_attributes, C)
    den = jnp.sum(inv_D_mat, axis=2)             # (num_attributes, C)
    energies = jnp.mean(num / den, axis=1)       # (num_attributes,)
    return energies

# We will vmap over the attribute dimension
euclidean_distances_batched = jax.vmap(euclidean_distances_jax, in_axes=0, out_axes=0)
cosine_distances_batched = jax.vmap(cosine_distances_jax, in_axes=0, out_axes=0)

if __name__ == '__main__':

    energy = {m:[] for m in models}

    for m in models:
        print("MODEL:", m)
        path = os.path.join(embeddings_path, m)
        files = [f for f in os.listdir(path) if f.endswith(".npy")]
        for p in tqdm(files):
            emb = np.load(os.path.join(path, p)).astype(np.float32)
            emb_jax = jnp.array(emb)
            en = compute_energy_for_identity(emb_jax, m)
            energy[m].append(np.array(en))
