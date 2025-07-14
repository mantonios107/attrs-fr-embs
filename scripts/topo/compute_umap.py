## Script for computing the UMAP projection of the embedding space 
## INPUTS
# --dataset: the name of the dataset 
# --method: the type of notebooks to perform. In particular
#              . "whole" to compute the UMAP projection of the whole dataset
#              . "barycenters" to compute the UMAP projection of the pc given by the identities' barycenters
## OUTPUT
# The script saves the projections in the ./results/ folder

import argparse
import numpy as np
import pandas as pd
import umap
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute UMAP of a given point cloud"
    )
    parser.add_argument("--dataset", required=True, type=str, default = "CelebA")
    parser.add_argument("--method", required=True, type=str, default = "whole")
    parser.add_argument("--neighbors", required=False, type=int, default = 900)
    args = parser.parse_args()

    dataset = args.dataset
    method = args.method
    neighbors = args.neighbors

    if dataset not in ["CelebA"]:
        raise Exception("Dataset name not recognized")
    if method not in ["whole","barycenters"]:
        raise Exception("Method not recognized")
    
    metadata = pd.read_parquet(f"../embeddings/{dataset}/df_dataset.parquet")
    metadata_identities = pd.read_parquet(f"../embeddings/{dataset}/df_dataset_identities.parquet")
    metadata_attributes = pd.read_parquet(f"../embeddings/{dataset}/df_dataset_attributes.parquet")
    
    MODEL_NAMES = ["retina_facenet","retina_arcface"]
    MODEL_METRIC = ["euclidean", "cosine"]
   
    embeddings = []
    for m, model_name in enumerate(MODEL_NAMES):
        data_npz = np.load(f"../embeddings/{dataset}/embeddings_{model_name}.npz")
        embeddings.append(data_npz['a'])

    UMAP_projections = []

    match method:
        case "whole":
            print("Computing UMAP on the whole dataset")
            for m,model_name in enumerate(MODEL_NAMES):
                print(model_name)
                proj = umap.UMAP(metric = MODEL_METRIC[m], n_neighbors = neighbors).fit_transform(
                    embeddings[m][
                        metadata.loc[metadata[f"keep_{model_name}"]==True].index,:
                        ])
                UMAP_projections.append(proj)
        
        case "barycenters":
            print("Computing the barycenters")
            datas_barycenters = []
            identities_ids = []
            for m, model_name in enumerate(MODEL_NAMES):
                identities_ids.append(np.sort(metadata.loc[metadata[f"keep_{model_name}"]==True]["identity"].unique()))
                datas_barycenters.append([])
                for n, identity in enumerate(identities_ids[-1]):
                    datas_barycenters[-1].append(embeddings[m][metadata.loc[
                    (metadata[f"keep_{model_name}"]==True)&(metadata["identity"]==identity)
                    ].index,:].mean(axis = 0))

            # Convert to np.array
            datas_barycenters = [np.array(datas_barycenters[m]) for m in range(len(MODEL_NAMES))]

            print("Computing UMAP on the barycenters of the identities")
        
            for m,model_name in enumerate(MODEL_NAMES):
                print(model_name)
                proj = umap.UMAP(metric = MODEL_METRIC[m], n_neighbors = neighbors).fit_transform(
                    datas_barycenters[m])
                UMAP_projections.append(proj)


    with open(f'results/{dataset}_UMAP_projections_{method}_nn{neighbors}.pickle', 'wb') as handle:
        pickle.dump(UMAP_projections, handle, protocol=pickle.HIGHEST_PROTOCOL)
