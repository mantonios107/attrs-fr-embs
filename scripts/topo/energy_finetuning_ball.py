import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, paired_cosine_distances
from sklearn.neighbors import kneighbors_graph, RadiusNeighborsTransformer

import os, sys
import pickle
from scipy import stats
import random

from scipy.sparse import csr_matrix


def subsample_sparse_matrix(matrix, n):
    """
    Subsample nonzero elements in each row of a sparse matrix so that each row has at most n nonzero elements.
    
    Args:
        matrix (csr_matrix): Input sparse matrix in CSR format.
        n (int): Maximum number of nonzero elements allowed per row.
    
    Returns:
        csr_matrix: Subsampled sparse matrix.
    """
    if not isinstance(matrix, csr_matrix):
        raise ValueError("Input matrix must be a CSR sparse matrix.")

    # Create data for the subsampled matrix
    data = []
    indices = []
    indptr = [0]

    for i in range(matrix.shape[0]):
        # Get the start and end index for the row in data/indices
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]
        
        # Subsample nonzero elements if they exceed n
        if len(row_data) > n:
            sampled_indices = np.random.choice(len(row_data), size=n, replace=False)
            row_data = row_data[sampled_indices]
            row_indices = row_indices[sampled_indices]

        # Append the subsampled data and indices
        data.extend(row_data)
        indices.extend(row_indices)
        indptr.append(len(data))

    # Construct the new sparse matrix
    return csr_matrix((data, indices, indptr), shape=matrix.shape)


def compute_energy(epsilon=[0.1], max_curves =2*10**4, subsample = None, finetuning_path = None):
    
    finetunings = os.listdir(finetuning_path)
    finetunings = [f for f in finetunings if (os.path.isdir(os.path.join(finetuning_path,f))) and (f != "distances_interAttributes")]


    attributes = ['0','1','2','3','4','brightness', 'contrast', 'hue']
    features = ["Head angle", "Age", "Hair color", "Illumination", "Expression"] + ['Brightness', 'Contrast', 'Hue']


    # Precompute necessary quantities
    metadata = pd.read_csv("../../data/generated_synthetic/gan_control/df_param_step.csv") 
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
    
    folders = [f for f in os.listdir(finetuning_path) if f[0] != "."]
    #folders = ["iresnet100"]

    energy_fine = {f:{a:{e:[] for e in epsilon} for a in features} for f in folders}
    
    for folder in folders:
        print(f"FOLDER: {folder}")
        embeddings_path = os.path.join(finetuning_path,folder)
        for p in tqdm(os.listdir(embeddings_path+"/")): # iterate over identities
            if p.split(".")[1] == "npy":
                # Load embeddings
                embeddings = np.load(os.path.join(embeddings_path+"/",p))
                
                # Estimate average distance
                ids1 = np.random.randint(0,embeddings.shape[0],10000)
                ids2 = np.random.randint(0,embeddings.shape[0],10000)
                
                d_avg = np.mean(paired_cosine_distances(embeddings[ids1,:],embeddings[ids2,:]))
                
                for idx_dim,dim in enumerate(attributes):
                    embs_pc = embeddings[adfs_1[dim],:] # point cloud
                                        
                    n_points = embs_pc.shape[0]
                
                    vf_pc = embeddings[adfs_2[dim],:] - embeddings[adfs_0[dim],:] # vector field (3 point stencil)
                    #vf_pc = (vf_pc.T/np.linalg.norm(vf_pc, axis = 1)).T # normalize vector field
                            
                    for eps in epsilon:
                        knng = RadiusNeighborsTransformer(radius=eps*d_avg, mode='connectivity', metric = "cosine").fit_transform(embs_pc)
                
                        if subsample is not None:
                            knng = subsample_sparse_matrix(knng,subsample)

                        a,b = knng.nonzero()
                        un,counts = np.unique(a,return_counts=True)
                        dists_vf = paired_cosine_distances(vf_pc[a,:], vf_pc[b,:])
                        #print(un)
                        point_energy = np.zeros(n_points)

                        lim0 = 0
                        lim1 = 0
                        for point in range(n_points):
                            lim1 += counts[point]
                            point_energy[point] = np.mean(dists_vf[lim0:lim1])
                            lim0 = lim1

                        ene = np.mean(point_energy)
                    
                        energy_fine[folder][features[idx_dim]][eps].append(ene)

    return energy_fine

if __name__ == "__main__":
    savepath = "./saved_energies_ball/"

    #    input = "[2,3,4,5]"
    epsilon = list(map(float, sys.argv[1].strip('[]').split(',')))
    
    max_curves = int(sys.argv[2])
    if len(sys.argv) > 3:
        subsample = int(sys.argv[3])
    else:
        subsample = None
    if len(sys.argv) > 4:
        finetuning_path = sys.argv[4]
    else:
        #finetuning_path = "../../data/generated_synthetic/gan_control/embeddings/finetuning_20250117/adaface/best"
        finetuning_path = "../../data/generated_synthetic/gan_control/embeddings/vanilla"
    if len(sys.argv) > 5:
        savename = sys.argv[5]
    else:
        # savename = f'energies_adaface_eps{np.round(epsilon,2)}_curves{max_curves}.pickle'
        savename = f'energies_both_baseline_curves{max_curves}.pickle'
    
    energy_fine = compute_energy(epsilon,max_curves, subsample, finetuning_path =finetuning_path)
        
    with open(os.path.join(savepath,savename), 'wb') as handle:
        pickle.dump(energy_fine, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
    
# chmod u+x run_scripts.sh
# ./job.sh