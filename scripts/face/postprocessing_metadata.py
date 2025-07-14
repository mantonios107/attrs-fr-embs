## Script to create summaries of metadata for identities and attributes

from sklearn.neighbors import KernelDensity
import argparse, os
import pandas as pd
import numpy as np
import scipy

### GLOBALS ###

DATA_PATH = "../../data"

L_ENRICHED_ATTRIBUTES = ["rgb_r", "rgb_g", "rgb_b", "hsv_h", "hsv_s", "hsv_v"]

L_CELEBA_ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows',
       'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
       'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
       'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
       'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']


### FUNCTIONS ###

def get_datasetName(path):
    """
    path is of the form */data/dataset_name/*
    """
    while os.path.basename(os.path.dirname(path))!="data":
        path = os.path.dirname(path)
    dataset_name = os.path.basename(path)
    return dataset_name

def summarize_enrichedFeaturesByAttribute(df_dataset):
    return pd.DataFrame(df_dataset[L_ENRICHED_ATTRIBUTES].mean(), columns=["global_average"])

def summarize_enrichedFeaturesByIdentity(df_dataset):
    d_agg = {k:"mean" for k in L_ENRICHED_ATTRIBUTES}
    d_agg["identity"] = "size"
    df = df_dataset[["identity"]+L_ENRICHED_ATTRIBUTES].groupby("identity").agg(d_agg).rename(columns={"identity":"size"})
    df.columns = ["mean_" + x  if x!="size" else x for x in df.columns]
    return df

def compute_interEntropyByAttribute(df_mia, l_attributes):
    l_H = []
    n_states = 101
    for i, attribute in enumerate(l_attributes):
        arr = df_mia.loc[df_mia["attribute"]==attribute, "id_average"].values
        kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
        kde.fit(arr.reshape(-1,1))
        x = np.linspace(-1,1,n_states)
        logprob = kde.score_samples(x.reshape(-1,1))
        prob = np.exp(logprob)/(np.exp(logprob).sum())
        H = scipy.stats.entropy(prob)
        l_H.append(H)
    return pd.Series(l_H, index=l_attributes, name="inter_entropy")

def parse_metadataBoolean(df_dataset, df_attributes, df_identities, l_attributes):

    # creation of df_mia (mean attribute value by identity and attribute) and df_hia (entropy by identity and attribute)
    df_mia = df_dataset[l_attributes + ["identity"]].groupby("identity").agg("mean").reset_index()
    df_mia = df_mia.melt(id_vars=["identity"]).rename(columns={"variable":"attribute", "value":"id_average"})
    df_hia = df_dataset[l_attributes + ["identity"]].groupby("identity").agg(lambda u:  scipy.stats.entropy(np.unique(np.array(list(u)), return_counts=True)[1])).reset_index()
    df_hia = df_hia.melt(id_vars=["identity"]).rename(columns={"variable":"attribute", "value":"id_entropy"})

    # df_attributes
    df_attributes = pd.concat([df_attributes, pd.DataFrame(df_dataset[l_attributes].mean().rename("global_average"))])
    df_attributes = pd.merge(df_attributes
                         ,compute_interEntropyByAttribute(df_mia, l_attributes)
                         ,how="left", right_index=True, left_index=True)
    df_attributes = pd.merge(df_attributes
                         ,df_hia.groupby("attribute")["id_entropy"].mean().rename("intra_entropy")
                         ,how="left", right_index=True, left_index=True)
    
    # df_identities
    df = df_hia.pivot(index="identity", columns=["attribute"])
    df.columns = ["entropy_" + x for x in df.droplevel(0, axis=1).columns]
    df["average_entropy"] = df.values.mean(axis=1)
    df_identities = pd.merge(df_identities, df,
         how="left", left_index=True, right_index=True)
    df = df_mia.pivot(index="identity", columns=["attribute"])
    df.columns = ["mean_" + x for x in df.droplevel(0, axis=1).columns]
    df_identities = pd.merge(df_identities, df,
             how="left", left_index=True, right_index=True)
    return df_attributes, df_identities


### MAIN ###

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="computes summary statistics of images")
    parser.add_argument("--df_parquet", type=str, required=True)
    parser.add_argument("--high_quality_subset", type=str, required=False, default=None)
    args = parser.parse_args()
    dataset_name = get_datasetName(args.df_parquet)
    print(f"dataset_name={dataset_name}")
    df_dataset = pd.read_parquet(args.df_parquet)
    
    if args.high_quality_subset is not None:
        df_dataset = df_dataset.loc[df_dataset[f"keep_embeddings_retina_{args.high_quality_subset}"]]
        print(f"Using keep_embeddings_retina_{args.high_quality_subset} column as a high quality subset")
    
    df_attributes = summarize_enrichedFeaturesByAttribute(df_dataset)
    df_identities = summarize_enrichedFeaturesByIdentity(df_dataset)
    
    if dataset_name=="CelebA":
        df_attributes, df_identities = parse_metadataBoolean(df_dataset, df_attributes, df_identities, L_CELEBA_ATTRIBUTES)
    if dataset_name=="lfw":
        df_dataset["male"] = df_dataset["gender"].map({"male":True, "female":False}).apply(lambda l: l if not np.isnan(l) else np.random.choice([True, False]))
        df_attributes, df_identities = parse_metadataBoolean(df_dataset, df_attributes, df_identities, ["male"])
        
    
    # save
    df_attributes.to_parquet(f"{DATA_PATH}/{dataset_name}/df_dataset_attributes.parquet")
    df_identities.to_parquet(f"{DATA_PATH}/{dataset_name}/df_dataset_identities.parquet")