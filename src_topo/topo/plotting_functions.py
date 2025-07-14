import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL_NAMES = ["retina_facenet","retina_arcface"]
cols = ['#005f73', '#0a9396', '#ca6702','#ffc300']

def plot_umap(dataset,method,neighbors,attribute = None, s = 3, frac = 1):
    ## INPUTS
    # dataset: str, dataset name
    # method: str, "whole" or "barycenters"
    # neighbors: int, the number of neighbors used for UMAP
    # attribute: str, if not None color the points according to said attribute
    # s: int, point size
    # frac: float, the fraction of points to dislay (speeds up the computation)

    if dataset not in ["CelebA"]:
        raise Exception("Dataset name not recognized")
    if method not in ["whole","barycenters"]:
        raise Exception("Method not recognized")

    with open(f'../results/{dataset}_UMAP_projections_{method}_nn{neighbors}.pickle', 'rb') as handle:
        UMAP_projections = pickle.load(handle)

    metadata = pd.read_parquet(f"../../embeddings/{dataset}/df_dataset.parquet")
    
    samples = [0,0]
    samples[0] = np.random.choice(UMAP_projections[0].shape[0], int(UMAP_projections[0].shape[0]*frac), replace=False)
    samples[1] = np.random.choice(UMAP_projections[1].shape[0], int(UMAP_projections[1].shape[0]*frac), replace=False)

    match method:
        case "whole":
            fig = make_subplots(rows=1, cols=2, subplot_titles=MODEL_NAMES)
            if attribute is None:
                for i,proj in enumerate(UMAP_projections):
                    ids = metadata.loc[(metadata[f"keep_{MODEL_NAMES[i]}"]==True)].index
                    fig.add_trace(go.Scatter(x=proj[samples[i],0],y=proj[samples[i],1],mode='markers', 
                                            marker = dict(size= 2),
                                            customdata=ids[samples[i]],
                                            hovertemplate='image index:%{customdata}'
                                            ), row=i//2+1, col=i%2+1)
                            
                fig.update_layout(width=1000,height = 500)
            else:
                for i,proj in enumerate(UMAP_projections):
                    ids = metadata.loc[(metadata[f"keep_{MODEL_NAMES[i]}"]==True)].index
                    feat = np.array(metadata[attribute][ids])
                    fig.add_trace(go.Scatter(x=proj[samples[i],0],y=proj[samples[i],1],mode='markers',
                                            marker = dict(size= 2, 
                                                        color = feat[samples[i]],
                                                        colorscale = "Bluered_r",
                                                        opacity = 0.6),
                                            customdata= ids[samples[i]],
                                            hovertemplate='image index:%{customdata}'
                                            ), row=i//2+1, col=i%2+1)

                    fig.update_layout(width=1000,height = 500,title = attribute)

        
        case "barycenters":
            fig = make_subplots(rows=1, cols=2, subplot_titles=MODEL_NAMES)
            if attribute is None:
                for i,proj in enumerate(UMAP_projections):
                    identities_ids = np.sort(metadata.loc[metadata[f"keep_{MODEL_NAMES[i]}"]==True]["identity"].unique())
                    fig.add_trace(go.Scatter(x=proj[:,0],y=proj[:,1],mode='markers', marker = dict(size= s),
                                customdata=identities_ids,
                                hovertemplate='identity index:%{customdata}'
                                ), row=i//2+1, col=i%2+1)
                
                fig.update_layout(width=1000,height = 500)
            else:
                metadata_identities = pd.read_parquet(f"../../embeddings/{dataset}/df_dataset_identities.parquet")
                for i,proj in enumerate(UMAP_projections):
                    identities_ids = np.sort(metadata.loc[metadata[f"keep_{MODEL_NAMES[i]}"]==True]["identity"].unique())
                    feat = np.array(metadata_identities["avg_"+attribute][identities_ids])
                    fig.add_trace(go.Scatter(x=proj[:,0],y=proj[:,1], mode='markers',
                                            marker = dict(size= s, color = feat,
                                                            colorscale = "Bluered_r",
                                                            opacity = 0.6),
                                            ), row=i//2+1, col=i%2+1)

                fig.update_layout(width=1000,height = 500,title = "avg_"+attribute)

    return fig
    

def plot_PH(dataset,method,maxdim, attribute = "Male"):
    ## INPUTS
    # dataset: str, dataset name
    # method: str, "whole" or "barycenters"
    # maxdim: int, the maximum homology dimension computed

    if dataset not in ["CelebA"]:
        raise Exception("Dataset name not recognized")
    if method not in ["whole","barycenters","identities","attribute","barycenters_attribute"]:
        raise Exception("Method not recognized")
    
    if method in ["attribute","barycenters_attribute"]:
        with open(f'../results/{dataset}_PH_{method}_{attribute}_maxdim{maxdim}.pickle', 'rb') as handle:
            results = pickle.load(handle)
    else:
        with open(f'../results/{dataset}_PH_{method}_maxdim{maxdim}.pickle', 'rb') as handle:
            results = pickle.load(handle)

    figures = results["figures"]
    pers_diagrams = results["pers_diagrams"]
    homology_dimensions = [i for i in range(maxdim+1)]

    match method:
        case "whole" | "barycenters":
            fig = make_subplots(rows=1, cols=len(MODEL_NAMES), subplot_titles=MODEL_NAMES)

            ## Add each figure to a subplot
            for i, fig_obj in enumerate(figures, start=1):
                for trace in fig_obj.data:
                    fig.add_trace(trace, row=1, col=i)

            for j in homology_dimensions:
                fig.update_traces(
                    marker=dict(color=cols[j]),
                    selector=dict(type="scatter", name=f"H{j}"))

            fig.update_layout(width=800,height = 400)

        case "attribute" | "barycenters_attribute":
            titles = [f"{model_name} {attribute} = {i}" for i in [-1,1] for model_name in MODEL_NAMES]
            fig = make_subplots(rows=2, cols=len(MODEL_NAMES), subplot_titles=titles)

            ## Add each figure to a subplot
            for m in range(len(MODEL_NAMES)):
                data = figures[m].data
                for i, trace in enumerate(data[0:len(data)//2]):
                    fig.add_trace(trace, row=1, col=m+1)
                for i, trace in enumerate(data[len(data)//2:]):
                    fig.add_trace(trace, row=2, col=m+1)

                for j in homology_dimensions:
                    fig.update_traces(
                        marker=dict(color=cols[j]),
                        selector=dict(type="scatter", name=f"H{j}"))

            fig.update_layout(width=800,height = 800)


    return fig, pers_diagrams
    