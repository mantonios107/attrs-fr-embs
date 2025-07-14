## script to create a file df_dataset.parquet that will be used to store paths of image and metadata on images.
## df_dataset.parquet will be used for running models (detection and recognition) and for several analyses

import argparse
import numpy as np
import pandas as pd
import glob
import os, sys
sys.path.append("../../src_face")
import face.utils

DATA_PATH = "../../data"

def parse_lfw():
    """
    Creates df_dataset_full and df_dataset parquet files
    """
    l_paths = [x for x in glob.glob(f"{DATA_PATH}/lfw/lfw"+"/*/*")]
    l_identity = [os.path.basename(os.path.dirname(x)) for x in l_paths]
    df_dataset = pd.DataFrame({"path":l_paths
                               ,"identity":l_identity})
    assert len(df_dataset) == 13233  # number of images in lfw dataset
    df_males = pd.read_csv(f"{DATA_PATH}/lfw/male_names.txt", sep="\t", header=None, names=["name"])
    df_males["gender"] = "male"
    df_females = pd.read_csv(f"{DATA_PATH}/lfw/female_names.txt", sep="\t", header=None, names=["name"])
    df_females["gender"] = "female"
    df = pd.concat([df_males, df_females], axis=0).sample(frac=1).drop_duplicates(subset="name")  # trying to correct identities that have different genders
    df_dataset["file_basename"] = df_dataset["path"].apply(os.path.basename)
    if "gender" in df_dataset.columns:
        l_cols_toDrop = ["gender"]
    else:
        l_cols_toDrop = []
    df_dataset = df_dataset.drop(columns=l_cols_toDrop).merge(df, how="left", left_on="file_basename", right_on="name").drop(columns=["name"])
    
    # sanity check: every identity has only 1 gender
    df = df_dataset.groupby(["identity", "gender"]).size().rename("count").reset_index()
    assert len(df) == df["identity"].nunique()
    return df_dataset

    # df_dataset.to_parquet(f"{DATA_PATH}/lfw/df_dataset_full.parquet")
    # df_dataset = filter_identityCardinal(df_dataset, min_images_per_identity)
    # df_dataset.to_parquet(f"{DATA_PATH}/lfw/df_dataset.parquet")

def parse_CelebA():
    """
    Creates df_dataset_full and df_dataset parquet files
    """
    path_identity = os.path.join(f"{DATA_PATH}/CelebA", *["Anno", "identity_CelebA.txt"])
    path_attr = os.path.join(f"{DATA_PATH}/CelebA", *["Anno", "list_attr_celeba.txt"])
    path_bbox = os.path.join(f"{DATA_PATH}/CelebA", *["Anno", "list_bbox_celeba.txt"])
    path_landmarks = os.path.join(f"{DATA_PATH}/CelebA", *["Anno", "list_landmarks_celeba.txt"])
    
    df_identities = pd.read_csv(path_identity, sep="\s+", header=None, names=["image", "identity"])
    df_attr = pd.read_csv(path_attr, skiprows=1, sep="\s+")
    df_bbox = pd.read_csv(path_bbox, skiprows=1, sep="\s+")
    df_landmarks = pd.read_csv(path_landmarks, skiprows=1, sep="\s+")
    
    df_dataset_full = df_identities.merge(df_attr.reset_index(), how="left", left_on="image", right_on="index").drop(columns=["index"]
                ).merge(df_bbox, how="left", left_on="image", right_on="image_id").drop(columns=["image_id"]
                ).merge(df_landmarks.reset_index(), how="left", left_on="image", right_on="index").drop(columns=["index"])
    df_dataset_full["path"] = df_dataset_full["image"].apply(lambda u: os.path.join(f"{DATA_PATH}/CelebA/CelebA", u))
    df_dataset_full = df_dataset_full.rename(columns={"image":"file_basename"})
    return df_dataset_full
    # df_dataset_full.to_parquet(f"{DATA_PATH}/CelebA/df_dataset_full.parquet")
    # df_dataset = filter_identityCardinal(df_dataset_full, min_images_per_identity)
    # df_dataset.to_parquet(f"{DATA_PATH}/CelebA/df_dataset.parquet")

def filter_identityCardinal(df, min_images_per_identity):
    s = df.groupby("identity").size()
    return df.loc[df["identity"].isin(s.loc[s>=min_images_per_identity].index)].copy().reset_index(drop=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is the description"
    )
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--min_images_per_identity", required=True, type=int)
    args = parser.parse_args()
    min_images_per_identity = args.min_images_per_identity
    dataset_name = args.dataset_name

    if dataset_name=="lfw":
        df_dataset_full = parse_lfw()
    elif dataset_name=="CelebA":
        df_dataset_full = parse_CelebA()
    df_dataset = filter_identityCardinal(df_dataset_full, min_images_per_identity)
    # save datasets
    df_dataset_full.to_parquet(f"{DATA_PATH}/{dataset_name}/df_dataset_full.parquet")
    df_dataset.to_parquet(f"{DATA_PATH}/{dataset_name}/df_dataset.parquet")
    