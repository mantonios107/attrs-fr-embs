import os
import random
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np  # still used for some I/O operations if needed
from tqdm import tqdm

# Ensure GPU/TPU usage if available (JAX automatically picks device).
# Check what device is being used:
print("JAX is using device:", jax.devices())

embeddings_path = "../../data/embeddings/embeddings/"
models = ["facenet","arcface"]
attributes = ['0','1','2','3','4','brightness', 'contrast', 'hue']
max_curves = 5*10**3 # Number of curves to be sampled

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

# Compute the indices of the embeddings
adfs_0 = {dim: adfs[dim][adfs[dim]["param_step"]==0]["index"].values for dim in attributes}
adfs_1 = {dim: adfs[dim][adfs[dim]["param_step"]==1]["index"].values for dim in attributes}
adfs_2 = {dim: adfs[dim][adfs[dim]["param_step"]==2]["index"].values for dim in attributes}

# Convert to JAX arrays for faster indexing and slicing later
adfs_0_jax = {dim: jnp.array(adfs_0[dim]) for dim in attributes}
adfs_1_jax = {dim: jnp.array(adfs_1[dim]) for dim in attributes}
adfs_2_jax = {dim: jnp.array(adfs_2[dim]) for dim in attributes}

def euclidean_distances_jax(X):
    # X shape: (N, D)
    # Compute pairwise Euclidean distances: sqrt((x_i - x_j)^2)
    # A common trick: dist(i,j) = sqrt(||X[i]||^2 + ||X[j]||^2 - 2 X[i].dot(X[j]))
    norms = jnp.sum(X**2, axis=1, keepdims=True)
    D = jnp.sqrt(jnp.maximum(norms + norms.T - 2 * (X @ X.T), 0))
    return D

def cosine_distances_jax(X):
    # X shape: (N, D)
    # cosine_similarity = (X X^T) / (||X|| ||X||)
    # cosine_distance = 1 - cosine_similarity
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    sim = (X / norms) @ (X / norms).T
    return 1.0 - sim

@jax.jit
def compute_energy_for_identity(embeddings, model_type):
    # embeddings: jnp array of shape (M, d), M = total embeddings, d = embedding dimension

    energies = []
    for dim in attributes:
        # Extract relevant subsets
        embs_pc = embeddings[adfs_1_jax[dim], :]  # point cloud
        vf_pc = embeddings[adfs_2_jax[dim], :] - embeddings[adfs_0_jax[dim], :]  # vector field
        norms = jnp.linalg.norm(vf_pc, axis=1, keepdims=True)
        vf_pc = vf_pc / norms  # normalize vector field

        if model_type == "facenet":
            D_mat = euclidean_distances_jax(embs_pc)
        else:
            D_mat = cosine_distances_jax(embs_pc)

        # Replace zero distances with inf
        D_mat = jnp.where(D_mat == 0, jnp.inf, D_mat)
        inv_D_mat = 1.0/D_mat
        inv_D_mat = jnp.nan_to_num(inv_D_mat, posinf=0.0)

        D_mat_vf = cosine_distances_jax(vf_pc)

        num = jnp.sum(inv_D_mat * D_mat_vf, axis=1)
        den = jnp.sum(inv_D_mat, axis=1)
        en_dim = jnp.mean(num / den)
        energies.append(en_dim)

    return jnp.array(energies)

energy = {m:[] for m in models}

# Loading and processing loop
for m in models:
    print("MODEL: ", m)
    path = os.path.join(embeddings_path, m)
    files = [f for f in os.listdir(path) if f.endswith(".npy")]
    for p in tqdm(files):
        # Load embeddings as a float32 JAX array for efficiency
        emb = np.load(os.path.join(path, p)).astype(np.float32)
        emb_jax = jnp.array(emb)

        en = compute_energy_for_identity(emb_jax, m)
        # Convert back to host (if needed) to append to a Python list
        energy[m].append(np.array(en))
