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


def compute_correlation(max_curves =2*10**4, finetuning_path = None):
    
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
    #folders = ["vanilla"]
    
    corr_fine = {f:[] for f in folders}
    
    for folder in folders:
        print(f"FOLDER: {folder}")
        embeddings_path = os.path.join(finetuning_path,folder)
        for p in tqdm(os.listdir(embeddings_path+"/")): # iterate over identities
            if p.split(".")[1] == "npy":
                # Load embeddings
                embeddings = np.load(os.path.join(embeddings_path+"/",p))
                
                fields_list = []
                for idx_dim,dim in enumerate(attributes):
                    embs_pc = embeddings[adfs_1[dim],:] # point cloud
                    vf_pc = embeddings[adfs_2[dim],:] - embeddings[adfs_0[dim],:] # vector field (3 point stencil)
                    fields_list.append(vf_pc)
                
                Cdists = np.zeros((len(features),len(features)))
                for i in range(len(features)):
                    for j in range(len(features)):
                        Cdists[i,j] = np.mean(paired_cosine_distances(fields_list[i],fields_list[j]))
                    
                corr_fine[folder].append(Cdists)

    return corr_fine

if __name__ == "__main__":
    savepath = "./saved_energies_ball/"
    
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
        savename = f'correlation_arcface_finetuning_curves{max_curves}.pickle'
    
    corr_fine = compute_correlation(epsilon,max_curves, subsample, finetuning_path =finetuning_path)
        
    with open(os.path.join(savepath,savename), 'wb') as handle:
        pickle.dump(corr_fine, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
    
# chmod u+x run_scripts.sh
# ./job.sh