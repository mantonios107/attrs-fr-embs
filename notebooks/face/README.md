# Notebooks

- evaluate_embeddings_celeba.ipynb is an auxiliary notebook assessing the performance of recognition models on the binary classification task of predicting matching and unmatching pairs of faces taken from the LFW dataset.
- evaluate_embeddings_generated_gan_control.ipynb is an auxiliary notebook assessing the id retention when modifying an image with GAN-control. To do that, we measure the distances from pairs of images for the same person at different steps of modifications.
- experiment_distance_macroscale.ipynb is the macroscale experiment described in the paper. It takes as input a dataset preprocessed from CelebA. This notebook includes computations of the entropies, distance distributions in embedding space, KS-statistics, correlations and some visualizations.
