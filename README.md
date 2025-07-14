# Attributes Shape the Embedding Space of Face Recognition Models

This repository contains the official implementation of the [paper](https://openreview.net/forum?id=VY74pP1w93):

> **Attributes Shape the Embedding Space of Face Recognition Models**  
> Pierrick Leroy, Antonio Mastropietro, Marco Nurisso, Francesco Vaccarino  
> *Forty-second International Conference on Machine Learning (ICML)*

## Introduction

In this work, we investigate how facial attributes influence the embedding space of face recognition models, focusing on FaceNet, ArcFace, AdaFace. 
By analyzing the relationship between attributes and embeddings, we provide insights into the sensitivity and structure of various Face Recognition models. 
This repository includes the code and scripts to reproduce the experiments and results presented in the paper.

The embeddings were produced by different model from different repositories reported in the following table:
| Model      | Architecture | Metric    | Train Set | Images (M) | Source Repository            |
|------------|--------------|-----------|-----------|------------|------------------------------|
| FaceNet    | iResNetv1    | euclidean | VGGFace2  | 3.31       | davidsandberg/facenet        |
| ArcFace    | ResNet50     | cosine    | MS1MV3    | 5.18       | deepinsight/insightface      |
| ArcFace    | ResNet18     | cosine    | MS1MV3    | 5.18       | deepinsight/insightface      |
| AdaFace    | ResNet18     | cosine    | VGGFace2  | 3.31       | mk-minchul/AdaFace           |
| SphereFaceR| iResNet100   | cosine    | MS1       | 10         | ydwen/opensphere             |


### Macroscale analysis

The macroscale analysis depends on the CelebA dataset available [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
We process this dataset according to the pipeline described in src_face/README.md.
The raw LFW dataset is used as a sanity check but is not available anymore. An aligned version is available [here](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset).


### Microscale analysis

For the **microscale analysis**, we used [GanControl](https://arxiv.org/abs/2101.02477) to generate many small variations of fake individuals.
The script at *scripts/topo/energy_finetuning_jax.py* processes gan control generated images. 
In addition, you can use the notebook "notebooks/energy/energy_gancontrol.ipynb" to finetune the energy model on your own 


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
