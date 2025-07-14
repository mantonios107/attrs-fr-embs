## This scripts adds a boolean column to the df_dataset.parquet file which flags with True rows belonging to a high quality subset (exactly 1 face detected, not an outlier in its identity point cloud and belonging to an identity with above threshold number of photos)

import argparse
import pandas as pd
import numpy as np
import faiss, os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="computes summary statistics of images")
    parser.add_argument("--df_parquet", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--min_photos_by_id", type=int, required=True)
    parser.add_argument("--geom_n_neighbors", type=int, required=True)
    args = parser.parse_args()
    
    df_dataset = pd.read_parquet(args.df_parquet)
    a_embeddings = np.load(args.embeddings_path)["a"]
    filename_embeddings = os.path.basename(args.embeddings_path).split(".")[0]
    rec_model = filename_embeddings.split("_")[-1]
    MIN_NUMBER_PHOTO_BY_IDENTITY = args.min_photos_by_id
    N_NEAREST_NEIGHBORS = args.geom_n_neighbors
    assert len(a_embeddings)==len(df_dataset)
    print("Loaded metadata (df_dataset) and embeddings")
    print(f"""filename_embeddings={filename_embeddings}\nrec_model={rec_model}\nmin_photos_by_id={MIN_NUMBER_PHOTO_BY_IDENTITY}\ngeom_n_neighbors={N_NEAREST_NEIGHBORS}""")
    
    # default value
    df_dataset[f"keep_{filename_embeddings}"] = False
    
    # subset into df the dataset to restrict to images with 1 faces AND identities with at least MIN_NUMBER_PHOTO_BY_IDENTITY of such images
    mask_n_faces = df_dataset["retina_n_faces_detected"]==1
    s = df_dataset.loc[mask_n_faces].groupby("identity").size()
    a_identities = s.loc[s>=MIN_NUMBER_PHOTO_BY_IDENTITY].index.values
    mask_size = (df_dataset["identity"].isin(a_identities)) & (mask_n_faces)
    df = df_dataset.loc[mask_size]

    # N_NEAREST_NEIGHBORS cleaning
    for identity in df["identity"].unique():
        X_identity = a_embeddings[df[df["identity"]==identity].index.values]  # point cloud for this identity
        if rec_model == "arcface":
            index = faiss.IndexFlatIP(X_identity.shape[1])
            faiss.normalize_L2(X_identity)
            index.add(X_identity)
            similarities, ann = index.search(X_identity, len(X_identity))
            distances = 1-similarities
        if rec_model == "facenet":
            index = faiss.IndexFlatL2(X_identity.shape[1])
            index.add(X_identity)
            distances, ann = index.search(X_identity, len(X_identity))
        median = np.median(distances)  # Remark: possible to approximate and avoid computing the whole matrix by sampling pairs and computing distance
        a_outlier_scores = distances[:,N_NEAREST_NEIGHBORS]
        mask_distances = a_outlier_scores<median
        a_keep = df.loc[df["identity"]==identity].loc[mask_distances].index
        df_dataset.loc[a_keep, f"keep_{filename_embeddings}"] = True
    print(f"""{len(df_dataset)} total rows""")
    print(f"""{len(df)} rows with only 1 face and at least {MIN_NUMBER_PHOTO_BY_IDENTITY} photos per identity""")
    print(f"""{df_dataset[f"keep_{filename_embeddings}"].sum()} rows with point cloud outliers removed""")

    # post filtering: keep only indentities where there is at least MIN_NUMBER_PHOTO_BY_IDENTITY images left after removing geometric outliers
    s_post = df_dataset.loc[df_dataset[f"keep_{filename_embeddings}"],"identity"].value_counts()>=MIN_NUMBER_PHOTO_BY_IDENTITY
    mask_post = df_dataset["identity"].isin(s_post.loc[s_post].index)
    df_dataset[f"keep_{filename_embeddings}"] = df_dataset[f"keep_{filename_embeddings}"]&mask_post
    print(f"""{df_dataset[f"keep_{filename_embeddings}"].sum()} rows selected with still at least {MIN_NUMBER_PHOTO_BY_IDENTITY} photos per identity""")
    
    # save
    df_dataset.to_parquet(args.df_parquet)