## Script for computing the persistent homology dimension of the embedding space 
## INPUTS
# --dataset: the name of the dataset 
# --method: the type of analysis to perform. In particular
#              . "whole" to compute the PHD of the whole dataset
#              . "barycenters" to compute the PHD of the pc given by the identities' barycenters
#              . "identities" to compute the PHD of each identity pc 
#              . "attribute" to compute the PHD of the whole dataset when divided by attribute
## OUTPUT
# The script saves the diagrams and the figures in the ./results/ folder


import argparse
import numpy as np
import pandas as pd
import pickle
from topo.support_functions import *
from tqdm import tqdm

N_POINTS  = 9
N_RERUNS = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the persistent diagrams of a given point cloud"
    )
    parser.add_argument("--dataset", required=True, type=str, default = "CelebA")
    parser.add_argument("--method", required=True, type=str, default = "whole")
    parser.add_argument("--repetitions", required=False, type=int, default = 5)
    parser.add_argument("--attribute", required=False, type = str, default = "Male")
    args = parser.parse_args()

    dataset = args.dataset
    method = args.method
    nrep = args.repetitions
    attribute = args.attribute

    if dataset not in ["CelebA"]:
        raise Exception("Dataset name not recognized")
    if method not in ["whole","barycenters","identities","attribute","barycenters_attribute"]:
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

    dimensions = []

    match method:
        case "whole":
            # Returns a list of lists whose first dimension refers to the model, the second to the repetition
            print("Computing the PHD of the whole dataset")
            for m,model_name in enumerate(MODEL_NAMES):
                dimensions.append([])
                for n in range(nrep):
                    PHD_solver = PHD(alpha=1, metric=MODEL_METRIC[m], n_points=N_POINTS, n_reruns =  N_RERUNS)
                    dimensions[-1].append(PHD_solver.get_phd_single(input=embeddings[m]))
    
        case "barycenters":
            # Returns a list of lists whose first dimension refers to the model, the second to the repetition
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

            print("Computing the PHD of the barycenters of the identities")
            for m,model_name in enumerate(MODEL_NAMES):
                print(model_name)
                dimensions.append([])
                for n in tqdm(range(nrep)):
                    PHD_solver = PHD(alpha=1, metric=MODEL_METRIC[m],n_points=N_POINTS,n_reruns=N_RERUNS)
                    dimensions[-1].append(PHD_solver.get_phd_single(input=datas_barycenters[m]))

        case "identities":
            # Return a list of lists. First dimension: model, second dimension: identity, third dimension: repetition  
            print("Computing the PHD of the identities point clouds")
            for m, model_name in enumerate(MODEL_NAMES):
                print(model_name)
                dimensions.append([])
                identities_ids = np.sort(metadata.loc[metadata[f"keep_{model_name}"]==True]["identity"].unique())
                for id, identity in tqdm(enumerate(identities_ids)):
                    dimensions[-1].append([])
                    identity_pc =  embeddings[m][metadata.loc[
                    (metadata[f"keep_{model_name}"]==True)&(metadata["identity"]==identity)
                    ].index,:]
                    for n in range(nrep):
                        PHD_solver = PHD(alpha=1, metric=MODEL_METRIC[m],n_points=N_POINTS,n_reruns=N_RERUNS)
                        dimensions[-1][-1].append(PHD_solver.get_phd_single(input=identity_pc))
                

        case "attribute":
            print(f"Computing the PHD of the whole space divided by the binary attribute \"{attribute}\"")
            for m, model_name in enumerate(MODEL_NAMES):
                print(model_name)
                dimensions.append([[],[]])
                pc0 = embeddings[m][metadata.loc[
                    (metadata[f"keep_{model_name}"]==True)&(metadata[attribute]==-1)
                    ].index,:]
                pc1 = embeddings[m][metadata.loc[
                    (metadata[f"keep_{model_name}"]==True)&(metadata[attribute]==1)
                    ].index,:]
                for n in range(nrep):
                    PHD_solver = PHD(alpha=1, metric=MODEL_METRIC[m],n_points=N_POINTS,n_reruns=N_RERUNS)
                    dimensions[-1][0].append(PHD_solver.get_phd_single(input=pc0))
                for n in range(nrep):
                    PHD_solver = PHD(alpha=1, metric=MODEL_METRIC[m],n_points=N_POINTS,n_reruns=N_RERUNS)
                    dimensions[-1][1].append(PHD_solver.get_phd_single(input=pc1))
        
    if method in ["attribute"] :
        with open(f'results/{dataset}_PHdim_{method}_{attribute}_nrep{nrep}.pickle', 'wb') as handle:
            pickle.dump(dimensions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'results/{dataset}_PHdim_{method}_nrep{nrep}.pickle', 'wb') as handle:
            pickle.dump(dimensions, handle, protocol=pickle.HIGHEST_PROTOCOL)

