import os
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import scipy

class MetadataPostprocessor():
    
    """
    Creates identities and attributes summary dataframes
    Pre-filter df_dataset to compute the summaries on a subset
    """
    
    def __init__(self, df_dataset):
        self.l_enriched_attributes = ["rgb_r", "rgb_g", "rgb_b", "hsv_h", "hsv_s", "hsv_v"]
        self.l_celeba_attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows',
                                   'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                                   'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                                   'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
                                   'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                                   'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                                   'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                                   'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                                   'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        self.df_dataset = df_dataset
        self.dataset_name = self.get_datasetName()

    # Helper functions
        
    def get_datasetName(self):
        """
        Gets a sample path from self.df_dataset, the path must be of the form */data/dataset_name/*
        """
        path = self.df_dataset["path"].iloc[0]
        while os.path.basename(os.path.dirname(path))!="data":
            path = os.path.dirname(path)
        dataset_name = os.path.basename(path)
        return dataset_name

    # Metadata functions
    
    def get_metadataDataframes(self):
        df_attributes = self.summarize_enrichedFeaturesByAttribute()
        df_identities = self.summarize_enrichedFeaturesByIdentity()
        if self.dataset_name=="CelebA":
            df_attributes, df_identities = self.parse_metadataBoolean(df_attributes, df_identities, self.l_celeba_attributes)
        if self.dataset_name=="lfw":
            self.df_dataset["male"] = self.df_dataset["gender"].map({"male":True, "female":False}).apply(lambda l: l if not np.isnan(l) else np.random.choice([True, False]))
            df_attributes, df_identities = self.parse_metadataBoolean(df_attributes, df_identities, ["male"])
        return df_attributes, df_identities
    
    def summarize_enrichedFeaturesByAttribute(self):
        return pd.DataFrame(self.df_dataset[self.l_enriched_attributes].mean(), columns=["global_average"])        
        
    def summarize_enrichedFeaturesByIdentity(self):
        d_agg = {k:"mean" for k in self.l_enriched_attributes}
        d_agg["identity"] = "size"
        df = self.df_dataset[["identity"]+self.l_enriched_attributes].groupby("identity").agg(d_agg).rename(columns={"identity":"size"})
        df.columns = ["mean_" + x  if x!="size" else x for x in df.columns]
        return df

    def compute_interEntropyByAttribute(self, df_mia, l_attributes):
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

    def parse_metadataBoolean(self, df_attributes, df_identities, l_attributes):

        # creation of df_mia (mean attribute value by identity and attribute) and df_hia (entropy by identity and attribute)
        df_mia = self.df_dataset[l_attributes + ["identity"]].groupby("identity").agg("mean").reset_index()
        df_mia = df_mia.melt(id_vars=["identity"]).rename(columns={"variable":"attribute", "value":"id_average"})
        df_hia = self.df_dataset[l_attributes + ["identity"]].groupby("identity").agg(lambda u:  scipy.stats.entropy(np.unique(np.array(list(u)), return_counts=True)[1])).reset_index()
        df_hia = df_hia.melt(id_vars=["identity"]).rename(columns={"variable":"attribute", "value":"id_entropy"})

        # df_attributes
        df_attributes = pd.concat([df_attributes, pd.DataFrame(self.df_dataset[l_attributes].mean().rename("global_average"))])
        df_attributes = pd.merge(df_attributes
                             ,self.compute_interEntropyByAttribute(df_mia, l_attributes)
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