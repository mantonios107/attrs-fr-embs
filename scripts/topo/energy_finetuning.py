import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, paired_cosine_distances
from sklearn.neighbors import kneighbors_graph
import os, sys
import pickle
from scipy import stats
import random


#n_neighbors = 10

def compute_energy(n_neighbors=10):
    
    model = "arcface"
    max_curves = 2*10**4 # Number of curves to be sampled

    #finetuning_path = f"../../data/generated_synthetic/gan_control/embeddings/finetuning/{model}/"
    #finetuning_path = f"../../data/generated_synthetic/gan_control/embeddings/"
    #finetuning_path = "../../data/generated_synthetic/gan_control/embeddings/finetuning_20250110_10k_test_metrics/finetuning/arcface/"
    
    finetuning_path = "tda-face-recognition/data/generated_synthetic/gan_control/embeddings/finetuning_20250117/arcface/best"
    
    finetunings = os.listdir(finetuning_path)
    finetunings = [f for f in finetunings if os.path.isdir(os.path.join(finetuning_path,f)) ]


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
    
    print(adfs['0']["param_step"].unique())
    
    # Compute the indices of the emebeddings
    adfs_0 = {dim: adfs[dim][adfs[dim]["param_step"]==0]["index"].values for dim in attributes}
    adfs_1 = {dim: adfs[dim][adfs[dim]["param_step"]==1]["index"].values for dim in attributes}
    adfs_2 = {dim: adfs[dim][adfs[dim]["param_step"]==2]["index"].values for dim in attributes}


    folders = [f for f in os.listdir(finetuning_path) if f[0] != "."]
    #folders = ["vanilla"]

    energy_fine = {f:{a:[] for a in features} for f in folders}
    gravity_fine = {f:{a:[] for a in features} for f in folders}

    for folder in folders:
        print(f"FOLDER: {folder}")
        embeddings_path = os.path.join(finetuning_path,folder)
        for p in tqdm(os.listdir(embeddings_path+"/")): # iterate over identities
            if p.split(".")[1] == "npy":
                # Load embeddings
                embeddings = np.load(os.path.join(embeddings_path+"/",p))
                for idx_dim,dim in enumerate(attributes):
                    embs_pc = embeddings[adfs_1[dim],:] # point cloud
                    # Compute barycenter vector field
                    bary = np.mean(embs_pc,0,keepdims=True)
                    bary_vf = bary - embs_pc
                    bary_vf = (bary_vf.T/np.linalg.norm(bary_vf, axis = 1)).T
                
                    vf_pc = embeddings[adfs_2[dim],:] - embeddings[adfs_0[dim],:] # vector field (3 point stencil)
                    #vf_pc = (vf_pc.T/np.linalg.norm(vf_pc, axis = 1)).T # normalize vector field

                    # Compute distance matrix
                    #D_mat = cosine_distances(embs_pc)
                    knng = kneighbors_graph(embs_pc,n_neighbors,metric = "cosine")
                    a,b = knng.nonzero()

                    #D_mat_vf = cosine_distances(vf_pc) # Compute cosine distances between point clouds
                    
                    #ene = np.mean(D_mat_vf[a,b])
                    ene = np.mean(paired_cosine_distances(vf_pc[a,:], vf_pc[b,:]))
                    
                    grav = np.mean(np.sum(bary_vf*vf_pc,1))
                    
                    energy_fine[folder][features[idx_dim]].append(ene)
                    gravity_fine[folder][features[idx_dim]].append(grav)
                    #print(energy_fine)
                    
        save_dict = {'energy':energy_fine[folder], 'gravity':gravity_fine[folder]}
        with open(os.path.join(savepath,f'energies_new_arcface_{n_neighbors}neighbors_final_morecurves_{folder}.pickle'), 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved")

    return energy_fine, gravity_fine

if __name__ == "__main__":
    savepath = "./saved_energies/"
    n_neighbors = int(sys.argv[1])
    energy_fine, gravity_fine = compute_energy(n_neighbors)
    save_dict = {'energy':energy_fine, 'gravity':gravity_fine}
    with open(os.path.join(savepath,f'energies_new_arcface_{n_neighbors}neighbors_final_morecurves_baseline.pickle'), 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    