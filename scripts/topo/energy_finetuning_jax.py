import os
import random
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
import time

print("JAX is using device:", jax.devices())


model = "arcface"
max_curves = 3*10**3 # Number of curves to be sampled
n_neighbors = 10

finetuning_path = f"../../data/generated_synthetic/gan_control/embeddings/finetuning/{model}/"
finetunings = os.listdir(finetuning_path)
finetunings = [f for f in finetunings if os.path.isdir(os.path.join(finetuning_path,f)) ]


attributes = ['0','1','2','3','4','brightness', 'contrast', 'hue']
features = ["Head angle", "Age", "Hair color", "Illumination", "Expression"] + ['Brightness', 'Contrast', 'Hue']

# Load and preprocess metadata
metadata = pd.read_csv("../../data/generated_synthetic/gan_control/df_param_step.csv") 
metadata = metadata.assign(moving_dimension=lambda x: x['moving_dimension'].astype("str"))

metadatas = {dim: metadata.loc[metadata["moving_dimension"]==dim] for dim in attributes}
ids_curves = {dim: np.unique(metadatas[dim]["id_curve"].values) for dim in attributes}
for dim in ids_curves.keys():
    random.shuffle(ids_curves[dim])

adfs = {
    dim: metadatas[dim][metadatas[dim]["id_curve"].isin(ids_curves[dim][:max_curves])].sort_values("id_curve")
    for dim in attributes
}

adfs_0 = {dim: adfs[dim][adfs[dim]["param_step"]==0]["index"].values for dim in attributes}
adfs_1 = {dim: adfs[dim][adfs[dim]["param_step"]==1]["index"].values for dim in attributes}
adfs_2 = {dim: adfs[dim][adfs[dim]["param_step"]==2]["index"].values for dim in attributes}

# Convert to jnp arrays and stack along a new dimension for attributes
adfs_0_all = jnp.stack([jnp.array(adfs_0[dim]) for dim in attributes])
adfs_1_all = jnp.stack([jnp.array(adfs_1[dim]) for dim in attributes])
adfs_2_all = jnp.stack([jnp.array(adfs_2[dim]) for dim in attributes])

def euclidean_distances_jax(X):
    # X: (C, D)
    norms = jnp.sum(X**2, axis=1, keepdims=True)
    return jnp.sqrt(jnp.maximum(norms + norms.T - 2 * (X @ X.T), 0))

def cosine_distances_jax(X):
    # X: (C, D)
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    X_normed = X / norms
    sim = X_normed @ X_normed.T
    return 1.0 - sim

# Batch over attributes
euclidean_distances_batched = jax.vmap(euclidean_distances_jax, in_axes=0, out_axes=0)
cosine_distances_batched = jax.vmap(cosine_distances_jax, in_axes=0, out_axes=0)

@partial(jax.jit, static_argnames=['model_type'])
def compute_energy_for_identity(embeddings, model_type):
    # embeddings: (M, D)
    # Extract the point clouds and vector fields
    embs_pc = embeddings[adfs_1_all, :]  # (num_attributes, C, D)
    vf_pc = embeddings[adfs_2_all, :] - embeddings[adfs_0_all, :]  # (num_attributes, C, D)
    vf_norms = jnp.linalg.norm(vf_pc, axis=2, keepdims=True)
    vf_pc = vf_pc / vf_norms

    if model_type == "facenet":
        D_mat = euclidean_distances_batched(embs_pc)  # (num_attributes, C, C)
    else:
        D_mat = cosine_distances_batched(embs_pc)     # (num_attributes, C, C)

    # NEW VERSION with error
    # inv_D_mat = 1 / D_mat
    # inv_D_mat.at[jnp.isinf(inv_D_mat)].set(0)
    # OLD VERSION
    # D_mat = jnp.where(D_mat == 0, jnp.inf, D_mat)
    # inv_D_mat = jnp.nan_to_num(1.0/D_mat, posinf=0.0)
    D_mat_vf = cosine_distances_batched(vf_pc)     # (num_attributes, C, C)
    # num = jnp.sum(inv_D_mat * D_mat_vf, axis=2)
    # den = jnp.sum(inv_D_mat, axis=2)
    # energies = jnp.mean(num / den, axis=1)  # (num_attributes,)
    
    __, indices = jax.lax.top_k(-D_mat, n_neighbors+1) # (num_attributes, C, n_neighbors + 1)
    indices = indices[:,:,1:]      # (num_attributes, C, n_neighbors )
    
    # Compute the mean over the neighbors
    energies = jnp.mean(jnp.take_along_axis(D_mat_vf, indices, axis=-1), axis=(1,2))  # Shape: (num_attributes)
    
    return energies

##############################
# Parallel Processing Example:
##############################

folders = [f for f in os.listdir(finetuning_path) if f[0] != "."]
energy_fine = {f:{a:[] for a in features} for f in folders}

for folder in folders:
    print(f"FOLDER: {folder}")
    embeddings_path = os.path.join(finetuning_path,folder)
    t_start = time.time()
    files = [f for f in os.listdir(embeddings_path+"/") if f.endswith(".npy")]

    # Pre-load all embeddings into one large array
    # Assuming all identities have the same shape
    all_embeddings_list = []
    print('Loading the files')
    for i, p in enumerate(files):
        print(f"Loading embeddings: {100*(i+1)/len(files):.0f}%", end='\r')
        emb = np.load(os.path.join(embeddings_path + "/", p)).astype(np.float32)
        all_embeddings_list.append(emb)
    print(f'Done loading the {i+1} files  in time:', time.time() - t_start)
    # Stack into a single array: (num_identities, M, D)
    all_embeddings = np.stack(all_embeddings_list, axis=0)
    all_embeddings_jax = jnp.array(all_embeddings)

    # Vectorize over identities using vmap
    # in_axes=(0, None) means we apply `compute_energy_for_identity` to each identity (row of all_embeddings_jax)
    # separately, while model_type is constant (None) and not changing per identity
    batched_compute = jax.vmap(compute_energy_for_identity, in_axes=(0, None), out_axes=0)
    en_all = batched_compute(all_embeddings_jax, model)  # shape: (num_identities, num_attributes)

    # Convert result back to numpy for storage/analysis if needed
    en_all_np = np.array(en_all)
    # Now en_all_np[i] gives the energies for the i-th identity file.

    # If you want to store energies in the original structure:
    # energy = en_all_np.tolist()
    # energy now contains the energies for all identities at once.
    print(f"Time taken for finetuning {folder}: {time.time() - t_start:.2f} seconds")
    
    for i,f in enumerate(features):
        energy_fine[folder][f] = list(en_all_np[:,i])


with open(os.path.join(savepath,f'energies_jax_{n_neighbors}neighbors.pickle'), 'wb') as handle:
        pickle.dump(energy_fine, handle, protocol=pickle.HIGHEST_PROTOCOL)


# As suggested by Pierrick, for the first k-neighbours, we can use jax.lax.top_k
# https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.top_k.html
