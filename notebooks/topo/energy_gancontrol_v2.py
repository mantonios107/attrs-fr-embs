import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import os
import pickle
from scipy import stats
import random
import jax
from jax import numpy as jnp


embeddings_path = "../../data/embeddings/embeddings/"
models = [#"facenet",
          "arcface"]
attributes = ['0','1','2','3','4','brightness', 'contrast', 'hue']
max_curves = 5*10**3 # Number of curves to be sampled

energy = {m: [] for m in models}

# Precompute necessary quantities
metadata = pd.read_csv(os.path.join(embeddings_path,"df_param_step.csv"))

# Fix problem in metadata
metadata = metadata.assign(moving_dimension=lambda x: x['moving_dimension'].astype("str"))

# Filter the metadas for each attribute
metadatas = {dim: metadata.loc[metadata["moving_dimension"]==dim] for dim in attributes}

ids_curves = {dim:np.unique(metadatas[dim]["id_curve"].values) for dim in attributes}
for dim in ids_curves.keys():
    random.shuffle(ids_curves[dim])

# Sample the curves
adfs = {dim: metadatas[dim][metadatas[dim]["id_curve"].isin(ids_curves[dim][:max_curves])].sort_values("id_curve") for dim in attributes}

# Compute the indices of the emebeddings
# Shift the indices by -1 backward
adfs_0 = {dim: adfs[dim][adfs[dim]["param_step"]==0]["index"].values for dim in attributes}
# Indices of the embeddings
adfs_1 = {dim: adfs[dim][adfs[dim]["param_step"]==1]["index"].values for dim in attributes}
# Shift the indices by +1 forward
adfs_2 = {dim: adfs[dim][adfs[dim]["param_step"]==2]["index"].values for dim in attributes}

# From here
# https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
def cos_sim(A):
    similarity = np.dot(A, A.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return 1 - cosine.T * inv_mag

def simple_euclidean(X, Y):
    norms_XY = np.einsum("ij,ij->i", X, Y)
    np.sqrt(norms_XY, out=norms_XY)
    return norms_XY

def jax_simple_euclidean(X, Y):
    norms_XY = jnp.einsum("ij,ij->i", X, Y)
    return jnp.sqrt(norms_XY)

jax.vmap
def simple_cos(X, Y):
    norms_X = np.einsum("ij,ij->i", X, X)
    norms_Y = np.einsum("ij,ij->i", Y, Y)
    np.sqrt(norms_X, out=norms_X)
    np.sqrt(norms_Y, out=norms_Y)
    return 1 - (X @ Y.T) / (norms_X * norms_Y)

@jax.jit
def jax_simple_cos(X, Y):
    norms_X = jnp.einsum("ij,ij->i", X, X)
    norms_Y = jnp.einsum("ij,ij->i", Y, Y)
    return 1 - (X @ Y.T) / (jnp.sqrt(norms_X) * jnp.sqrt(norms_Y))

for m in models:
    print("MODEL: ", m)

    # iterate over identities
    for p in os.listdir(embeddings_path + m + "/"):

        if p.split(".")[1] == "npy":

            # Load embeddings
            embeddings = jnp.array(np.load(os.path.join(embeddings_path + m + "/", p)))
            en = []
            for dim in attributes:
                # point cloud
                embs_pc = embeddings[adfs_1[dim], :]

                # vector field (3 point stencil)
                vf_pc = embeddings[adfs_2[dim], :] - embeddings[adfs_0[dim], :]

                # normalize vector field
                vf_pc = (vf_pc.T / jnp.linalg.norm(vf_pc, axis=1)).T

                # Compute distance matrix
                if m == "facenet":
                    # Note that the default definitiona of euclidean_distances from sklearn
                    # uses the following formula
                    # ||x-y||^2 = <x-y, x-y> = <x, x> - 2 <x, y> + <y, y>
                    # < x, x > - 2 < x, y > + < y, y > =
                    #
                    D_mat = euclidean_distances(embs_pc)
                else:
                    D_mat = cosine_distances(embs_pc)
                    # D_mat = jax_simple_cos(embs_pc, embs_pc)#cosine_distances(embs_pc) # cos_sim(embs_pc)

                # QUESTION: WHAT IS THIS?
                # D_mat[D_mat == 0] = np.inf
                # inv_D_mat = np.nan_to_num(1 / D_mat, posinf=0)
                inv_D_mat = 1 / D_mat
                # inv_D_mat[np.isinf(inv_D_mat)] = 0
                inv_D_mat.at[jnp.isinf(inv_D_mat)].set(0)

                # Compute cosine distances between point clouds
                D_mat_vf = jax_simple_cos(embs_pc, embs_pc) # cosine_distances(vf_pc)

                en.append(jnp.mean(jnp.sum(inv_D_mat * D_mat_vf, 1) / jnp.sum(inv_D_mat, 1)))

            energy[m].append(en)
