# Attributes Shape the Embedding Space of Face Recognition Models

This repository contains the official implementation of the [paper](https://openreview.net/forum?id=VY74pP1w93):

> **Attributes Shape the Embedding Space of Face Recognition Models**  
> Pierrick Leroy, Antonio Mastropietro, Marco Nurisso, Francesco Vaccarino  
> *Forty-second International Conference on Machine Learning (ICML)*

# Table of contents

1. [Introduction](#introduction)  
   - [Summary](#summary)  
   - [Embeddings](#embeddings)  
2. [Macroscale analysis](#macroscale-analysis)  
   - [Datasets and preprocessing](#datasets-and-preprocessing)  
   - [Macroscale experiment](#macroscale-experiment)  
3. [Microscale analysis](#microscale-analysis)  
   - [Data generation with generative models](#data-generation-with-generative-models)  
   - [Microscale](#microscale)

# Introduction

## Summary

In this work, we investigate how attributes influence the embedding space of deep learning models.
While we focus on face recognition models like FaceNet, ArcFace and AdaFace, the methodology can be adapted to other domains. 
By analyzing the relationship between attributes (in our case, facial attributes) and embeddings, we provide insights into the sensitivity and structure of models. 
This repository includes the code and scripts to reproduce the experiments and results presented in the paper.

## Embeddings

In this work we analyze the *embedding space*.
To produce embeddings, a function mapping from the input space (face images) to the embedding space is needed.
The embeddings were produced by a variety of models coming from different repositories reported in the following table:
| Model      | Architecture | Metric    | Train Set | Images (M) | Source Repository            |
|------------|--------------|-----------|-----------|------------|------------------------------|
| FaceNet    | iResNetv1    | euclidean | VGGFace2  | 3.31       | [davidsandberg/facenet](https://github.com/davidsandberg/facenet)        |
| ArcFace    | ResNet50     | cosine    | MS1MV3    | 5.18       | [deepinsight/insightface](https://github.com/deepinsight/insightface)      |
| ArcFace    | ResNet18     | cosine    | MS1MV3    | 5.18       | [deepinsight/insightface](https://github.com/deepinsight/insightface)      |
| AdaFace    | ResNet18     | cosine    | VGGFace2  | 3.31       | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)           |
| SphereFaceR| iResNet100   | cosine    | MS1       | 10         | [ydwen/opensphere](https://github.com/ydwen/opensphere)             |


# Macroscale analysis

The macroscale analysis depends on the CelebA dataset available [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
The raw LFW dataset is used as a sanity check but is not available anymore. An aligned version is still available [here](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset).

The core of the macroscale analysis can be found in the notebook at notebooks/face/experiment_distance_macroscale.ipynb.\
It relies on preprocessing the CelebA dataset according to the pipeline described in scripts/face/README.md.

# Microscale analysis

For the **microscale analysis**, we used [GanControl](https://arxiv.org/abs/2101.02477) to generate many small variations of fake individuals on a structured lattice.
Each node of the lattice (corresponding to a face image) can then embedded with multiple face recognition models and a vector field can finally be derived (see the paper for more details).
This repository does not provide code for generating image, this part is done with GAN-control.
The script at *scripts/topo/energy_finetuning_jax.py* processes gan control generated images. 
In addition, you can see the notebook "notebooks/energy/energy_gancontrol.ipynb" to see how the energies are computed for two models.


## Requirements
To run the code, you need to install the following dependencies:

```bash
conda env create --name envname --file=environment_updated.yml
```

## Bibliography
If you use this code in your research, please cite our paper:

```
@inproceedings{leroyattributes,
  title={Attributes Shape the Embedding Space of Face Recognition Models},
  author={Leroy, Pierrick and Mastropietro, Antonio and Nurisso, Marco and Vaccarino, Francesco},
  booktitle={Forty-second International Conference on Machine Learning (ICML)},
  year={2025}
}
```
