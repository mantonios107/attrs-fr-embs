# Datasets preprocessing

These steps were taken to preprocess the CelebA and LFW datasets.

## 1. Load datasets

```console
python parse_dataset.py --dataset_name lfw --min_images_per_identity 0
python parse_dataset.py --dataset_name CelebA --min_images_per_identity 30
```

In the data/ folder, there should be folders with name of datasets.
Currently supported datasets are:
- lfw
- CelebA
  
In these folder, raw data and optionally attributes are stored according to the original dataset organisation. This organisation is dataset specific, for example CelebA/CelebA stores all images, whereas lfw/lfw contains folders of identities which themselves contain images.

### 1.1 lfw
The folder data/lfw/ contains
- lfw/ : folder containing identity folders
- male_names.txt :  text file storing gender attribute for each photo
- female_names.txt : same for females

Link to dataset [here](https://vis-www.cs.umass.edu/lfw/), we use the raw images (not aligned). The gender files from the paper ["AFIF4: Deep Gender Classification based on AdaBoost-based Fusion of Isolated Facial Features and Foggy Faces"](https://arxiv.org/abs/1706.04277) are on the same page under Resources -> LFW gender labeling

07/2025: LFW has been deprecated, a backup of already aligned images is still available [here](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

### 1.2 CelebA
The folder data/CelebA/ contains
- CelebA/: folder containing images
- Anno/: folder of metadata

Link to dataset [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), we use the raw images (not aligned). The metadata files are on the same page.

## 2. Run detection (RetinaFace)

```console
python detection.py --dataset_name lfw --path_model /home/disma_user/pleroy/.insightface/models/buffalo_l/det_10g.onnx
```

To run RetinaFace, it is necessary to have previously downloaded the onnx model det_10g.onnx. This model comes from insightface library (pip install or from git)

## 3. Run recognition models

For now, there are different scripts for each model because the requirements of each model are incompatible.

### 3.1 Run ArcFace

```console
python run_arcface.py --dataset_name lfw --path_cropped_imgs ../../data/lfw/detection_retina_112-112-3.npz --path_model /home/disma_user/pleroy/.insightface/models/buffalo_l/w600k_r50.onnx
python run_arcface.py --dataset_name CelebA --path_cropped_imgs ../../data/CelebA/detection_retina_112-112-3.npz --path_model /home/disma_user/pleroy/.insightface/models/buffalo_l/w600k_r50.onnx
```

ArcFace is run with the same requirements as RetinaFace since both are from insightface.
To run ArcFace, it is necessary to have downloaded the onnx model w600k_r50.onnx which comes from insightface library (pip install or build from git)


### 3.2 Run FaceNet
```console
python run_facenet.py --dataset_name lfw --path_cropped_imgs ../../data/lfw/detection_retina_112-112-3.npz
python run_facenet.py --dataset_name CelebA --path_cropped_imgs ../../data/CelebA/detection_retina_112-112-3.npz
```

The library [davidsandberg/facenet](https://github.com/davidsandberg/facenet) is used through the wrapper [keras-facenet](https://pypi.org/project/keras-facenet/). The model used has model_name 20180402-114759 which can be downloaded from this [link](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view?pli=1).
  
The model should be at ~/.keras-facenet/model_name i.e. ~/.keras-facenet/20180402-114759 should contains a file 20180402-114759-weights.h5. This file is to be built by the package wrapper keras-facenet.


## 4. Evaluate quality of embedding for a dataset (optional, notebook)

There is a notebook evaluate_embeddings.ipynb in notebooks/face that assess the quality of face embeddings.
The protocol mines matched pairs and unmatched pairs, which is inspired by the guidelines for evaluation from LFW dataset.

## 5. Enriching images
```console
python enrich_images_features.py --df_parquet ../../data/lfw/df_dataset.parquet --img_file_path ../../data/lfw/detection_retina_112-112-3.npz
python enrich_images_features.py --df_parquet ../../data/CelebA/df_dataset.parquet --img_file_path ../../data/CelebA/detection_retina_112-112-3.npz
```

The script enrich_images_features.py compute additional statistics, features on cropped images. These features can be computed for each image so it is not dataset specific.\
Features:
- RGB mean values
- HSV mean values

## 6. Postprocessing for data quality
```console
python postprocessing.py --df_parquet ../../data/lfw/df_dataset.parquet --embeddings_path ../../data/embeddings/lfw/embeddings_retina_facenet.npz --min_photos_by_id 25 --geom_n_neighbors 1
python postprocessing.py --df_parquet ../../data/CelebA/df_dataset.parquet --embeddings_path ../../data/embeddings/CelebA/embeddings_retina_facenet.npz --min_photos_by_id 25 --geom_n_neighbors 1
```

We propose to compute a high quality subset of photos of the dataset embedded. To improve quality we do filtering steps in this order:
1- filter out images with more than 1 face detected by the detection model (a significant part of errors are due to multiple faces on image)
2- filter out identities with less than a threshold of photos (--min_photos_by_id args)
3- geometric postprocessing to filter out outliers (methodology inspired by statistical removal techniques)
4- filter out identities that don't have enough photos (again, due to previous step)

Hyperparameters:
- --min_photos_by_id: minimum number of photos by identity
- --geom_n_neighbors: nearest neighbors to compute outlier score (default: 1, higher values are more restrictive). For every point in an identity point cloud, we compute the distance to the k-th neighbor, which we compare to the median distance in the identity point cloud.

## 7. Postprocessing metadata
```python
import pandas as pd
from src_face.face import utils
df_dataset = pd.read_parquet("../../data/CelebA/df_dataset.parquet")
df_dataset = df_dataset.loc[df_dataset["keep_embeddings_retina_arcface"]]  # optional, restrict df_dataset to high quality subset
metadata_processor = utils.MetadataPostprocessor(df_dataset)
df_attributes, df_identities = metadata_processor.get_metadataDataframes()
```

Summarize metadata at the identities and at the attributes levels. Two dataframes can be created (df_dataset_identities and df_dataset_attributes in the above snippet) with id and attribute respectively as index.
Restricting df_dataset to a high quality subset depends on the models because of the geometric postprocessing, this is why the model name is in the column "keep_embeddings_retina_arcface" used to filter.


## Flow chart
![Alt text](../../imgs/Flowchart_topoface_datagen.drawio.png?raw=true)
