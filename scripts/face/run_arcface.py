# !pip install -q insightface onnxruntime-gpu pyarrow

import argparse
import numpy as np
import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.app.common import Face

DATA_PATH = "../../data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is the description"
    )
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--path_cropped_imgs", required=True, type=str)
    parser.add_argument("--path_model", required=True, type=str)
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    path_cropped_imgs = args.path_cropped_imgs
    path_model = args.path_model
    
    a_aligned = np.load(path_cropped_imgs)["a"]
    rec_model = insightface.model_zoo.get_model(path_model)
    rec_model.prepare(ctx_id=0)  # 0 means using GPU, -1 is CPU

    l = []
    for i, img in enumerate(a_aligned):
        if i%100==0:
            print(f"{i} photos embedded")
        l.append(rec_model.get_feat(img).flatten())
    a_embeddings = np.stack(l)
    del l

    np.savez_compressed(f"{DATA_PATH}/embeddings/{dataset_name}/embeddings_retina_arcface.npz", a = a_embeddings)