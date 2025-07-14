# topoface

Experiments on the topology of the embedding space of Face Recognition Deep Learning models

The conda environment and packages are contained in `environment.yml` and `requirements.txt`.

The main scripts are `compute_PH.py` and `compute_UMAP.py` which can be called from command line and respectively compute the persistence diagrams and UMAP projections of the two models' embedding spaces.

The results are saved in the `results/` folder and can be called for visualization and analysis through the jupyter notebooks.

The folder `functions/` contains the functions necessary for performing the computations.


**Important:** the scripts take the dataset embeddings from a folder which should be put in the directory directly above `topoface/`. 
The embeddings folder should be named `embeddings` and should contain the subfolder `CelebA`.

Structure of the folder
- analysis
- - NOTEBOOKS
- functions
- results
- embeddings
- - CelebA
...

## Experiments

### UMAP projections of the embedding spaces
To generate the UMAP projections of the embedding spaces run the following:
- to project the whole space  `python compute_umap.py --dataset CelebA --method whole --neighbors [neighbors]` where `neighbors` specifies the number of neighbors considered by the algorithm;
- to project the space of barycenters of the identities point clouds  `python compute_umap.py --dataset CelebA --method barycenters --neighbors [neighbors]`.

The results obtained can be visualized in the `display_umap.ipynb` notebook. 
The point clouds obtained can be displayed with colors representing one particular attribute. 

### Mapper graph of the embedding spaces
A notebook to compute and display the mapper graph is found in `display_mapper_graph.ipynb`.

### PH of the embedding spaces
We have different ways of analyzing the PH of the embedding spaces:
- to compute the PH of the whole space `python compute_PH.py --dataset CelebA --method whole --max_homology [MH]` where `MH` is the max dimension of the homology to compute (usually MH = 1);
- to compute the PH of the barycenters of the identities point clouds `python compute_PH.py --dataset CelebA --method barycenters --max_homology [MH]`;
- to compute the PH of all the identities point clouds `python compute_PH.py --dataset CelebA --method identities --max_homology [MH]`;
- to compute the PH of the whole space divided by one attribute `python compute_PH.py --dataset CelebA --method attribute --max_homology [MH] --attribute [ATTR]` where `ATTR` is a valid attribute name;
- to compute the PH of the barycenters point cloud divided by one attribute `python compute_PH.py --dataset CelebA --method barycenters_attribute --max_homology [MH] --attribute [ATTR]` where `ATTR` is a valid attribute name.

The diagrams obtained can be visualized in the `display_PH.ipynb` notebook and analyzed in the `analyze_PH.ipynb`.

### PH dimension of the embedding spaces
Similarly to the case of persistent homology, we provide multiple ways to compute the persistent homology intrinsic dimension (PHD) of the embedding spaces:

- to compute the PHD of the whole space `python compute_PHdim.py --dataset CelebA --method whole --repetitions [N]` where `N` is the number of samples of the PHD to compute;
- to compute the PHD of the barycenters of the identities point clouds `python compute_PHdim.py --dataset CelebA --method barycenters --repetitions [N]`;
- to compute the PHD of all the identities point clouds separately `python compute_PHdim.py --dataset CelebA --method identities--repetitions [N]`;
- to compute the PHD of the whole space divided by one attribute `python compute_PHdim.py --dataset CelebA --method attribute --repetitions [N] --attribute [ATTR]` where `ATTR` is a valid attribute name;
