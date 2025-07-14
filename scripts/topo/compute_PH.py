## Script for computing the persistence diagrams of some aspects of the embedding space 
## INPUTS
# --dataset: the name of the dataset 
# --method: the type of notebooks to perform. In particular
#              . "whole" to compute the PH of the whole dataset
#              . "barycenters" to compute the PH of the pc given by the identities' barycenters
#              . "identities" to compute the PH of each identity pc 
#              . "attribute" to compute the PH of the whole dataset when divided by attribute
#              . "barycenters_attribute" to compute the PH of the barycenters pc divided by attribute
## OUTPUT
# The script saves the diagrams and the figures in the ./results/ folder


import argparse
import numpy as np
import pandas as pd
import pickle
from topo.support_functions import compute_plot_diagrams
import pathlib

DATA_PATH = pathlib.Path("../../data/")
RESULTS_PATH = pathlib.Path("../../data/results/")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the persistent diagrams of a given point cloud"
    )
    parser.add_argument("--dataset", required=True, type=str, default="CelebA")
    parser.add_argument("--method", required=True, type=str, default="whole")
    parser.add_argument("--max_homology", required=False, type=int, default=1)
    parser.add_argument("--attribute", required=False, type=str, default="Male")
    return parser.parse_args()

def main(args):
    dataset = args.dataset
    method = args.method
    max_homology = args.max_homology
    attribute = args.attribute

    if dataset not in ["CelebA"]:
        raise Exception("Dataset name not recognized")
    if method not in ["whole", "barycenters", "identities", "attribute", "barycenters_attribute"]:
        raise Exception("Method not recognized")

    metadata = pd.read_parquet(DATA_PATH / f"embeddings/{dataset}/df_dataset.parquet")
    metadata_identities = pd.read_parquet(DATA_PATH / f"embeddings/{dataset}/df_dataset_identities.parquet")
    metadata_attributes = pd.read_parquet(DATA_PATH / f"embeddings/{dataset}/df_dataset_attributes.parquet")

    MODEL_NAMES = ["retina_facenet", "retina_arcface"]
    MODEL_METRIC = ["euclidean", "cosine"]

    homology_dimensions = [i for i in range(max_homology + 1)]

    embeddings = []
    for m, model_name in enumerate(MODEL_NAMES):
        data_npz = np.load((DATA_PATH / f"embeddings/{dataset}/embeddings_{model_name}.npz").as_posix())
        embeddings.append(data_npz['a'])

    pers_diagrams = []
    figures = []

    match method:
        case "whole":
            # Return a list containing the diagrams of the models
            print("Computing the PH of the whole dataset")
            for m, model_name in enumerate(MODEL_NAMES):
                print(model_name)
                p_diag, fig = compute_plot_diagrams(data=[embeddings[m]],
                                                    metric=MODEL_METRIC[m],
                                                    homology_dimensions=homology_dimensions,
                                                    rescale=False,
                                                    plot=True,
                                                    show=False,
                                                    )
                pers_diagrams.append(p_diag)
                figures.append(fig)

        case "barycenters":
            # Return a list containing the diagrams of the models
            print("Computing the barycenters")
            datas_barycenters = []
            identities_ids = []
            for m, model_name in enumerate(MODEL_NAMES):
                identities_ids.append(
                    np.sort(metadata.loc[metadata[f"keep_{model_name}"] == True]["identity"].unique()))
                datas_barycenters.append([])
                for n, identity in enumerate(identities_ids[-1]):
                    datas_barycenters[-1].append(embeddings[m][metadata.loc[
                                                                   (metadata[f"keep_{model_name}"] == True) & (
                                                                               metadata["identity"] == identity)
                                                                   ].index, :].mean(axis=0))

            # Convert to np.array
            datas_barycenters = [np.array(datas_barycenters[m]) for m in range(len(MODEL_NAMES))]

            print("Computing the PH of the barycenters of the identities")
            for m, model_name in enumerate(MODEL_NAMES):
                print(model_name)
                p_diag, fig = compute_plot_diagrams(data=[datas_barycenters[m]],
                                                    metric=MODEL_METRIC[m],
                                                    homology_dimensions=homology_dimensions,
                                                    rescale=False,
                                                    plot=True,
                                                    show=False,
                                                    )
                pers_diagrams.append(p_diag)
                figures.append(fig)

        case "identities":
            # Return a list containing the lists of diagrams for each identity, for each model
            print("Computing the PH of the identities point clouds")
            for m, model_name in enumerate(MODEL_NAMES):
                print(model_name)
                identities_ids = np.sort(metadata.loc[metadata[f"keep_{model_name}"] == True]["identity"].unique())
                identity_pcs = []
                for n, identity in enumerate(identities_ids):
                    identity_pcs.append(embeddings[m][metadata.loc[
                                                          (metadata[f"keep_{model_name}"] == True) & (
                                                                      metadata["identity"] == identity)
                                                          ].index, :])
                p_diag, fig = compute_plot_diagrams(data=identity_pcs,
                                                    metric=MODEL_METRIC[m],
                                                    homology_dimensions=homology_dimensions,
                                                    rescale=False,
                                                    plot=False,
                                                    show=False,
                                                    )
                pers_diagrams.append(p_diag)
                figures.append(fig)

        case "attribute":
            print(f"Computing the PH of the whole space divided by the binary attribute \"{attribute}\"")
            for m, model_name in enumerate(MODEL_NAMES):
                print(model_name)
                pc0 = embeddings[m][metadata.loc[
                                        (metadata[f"keep_{model_name}"] == True) & (metadata[attribute] == -1)
                                        ].index, :]
                pc1 = embeddings[m][metadata.loc[
                                        (metadata[f"keep_{model_name}"] == True) & (metadata[attribute] == 1)
                                        ].index, :]
                p_diag, fig = compute_plot_diagrams(data=[pc0, pc1],
                                                    metric=MODEL_METRIC[m],
                                                    homology_dimensions=homology_dimensions,
                                                    rescale=False,
                                                    plot=True,
                                                    show=False,
                                                    )
                pers_diagrams.append(p_diag)
                figures.append(fig)

        case "barycenters_attribute":
            print("Computing the barycenters")
            datas_barycenters = []
            identities_ids = []
            for m, model_name in enumerate(MODEL_NAMES):
                identities_ids.append(
                    np.sort(metadata.loc[metadata[f"keep_{model_name}"] == True]["identity"].unique()))
                datas_barycenters.append([])
                for n, identity in enumerate(identities_ids[-1]):
                    datas_barycenters[-1].append(embeddings[m][metadata.loc[
                                                                   (metadata[f"keep_{model_name}"] == True) & (
                                                                               metadata["identity"] == identity)
                                                                   ].index, :].mean(axis=0))

            # Convert to np.array
            datas_barycenters = [np.array(datas_barycenters[m]) for m in range(len(MODEL_NAMES))]

            print(f"Computing the PH of the barycenters point cloud divided by the binary attribute \"{attribute}\"")
            for m, model_name in enumerate(MODEL_NAMES):
                print(model_name)
                feat = np.array(metadata_identities["avg_" + attribute][identities_ids[m]])
                pc0 = datas_barycenters[m][feat < 0, :]
                pc1 = datas_barycenters[m][feat > 0, :]
                p_diag, fig = compute_plot_diagrams(data=[pc0, pc1],
                                                    metric=MODEL_METRIC[m],
                                                    homology_dimensions=homology_dimensions,
                                                    rescale=False,
                                                    plot=True,
                                                    show=False,
                                                    )
                pers_diagrams.append(p_diag)
                figures.append(fig)

    output = {"figures": figures, "pers_diagrams": pers_diagrams}
    if method in ["attribute", "barycenters_attribute"]:
        save_path = RESULTS_PATH / f"{dataset}_PH_{method}_{attribute}_maxdim{max_homology}.pickle"
        with save_path.open('wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        save_path = RESULTS_PATH / f"{dataset}_PH_{method}_maxdim{max_homology}.pickle"
        with save_path.open('wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    args = parse_args()
    main(args)
