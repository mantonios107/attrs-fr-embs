# ## this script gets the df_dataset.parquet file and run retina face on each image in the path column
# ## stores number of faces detected in df_dataset.parquet

import argparse
from PIL import Image
import pickle, os
import datetime
import numpy as np
import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.app.common import Face
import pandas as pd

DATA_PATH = "../../data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is the description"
    )
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--path_model", required=True, type=str)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    path_model = args.path_model

    df_dataset = pd.read_parquet(f"{DATA_PATH}/{dataset_name}/df_dataset.parquet")
    print(f"Loading {dataset_name} ... {len(df_dataset)} lines loaded")
    print(f"ort.get_device() : {ort.get_device()}")
    print(f"ort.__version__ : {ort.__version__}")
    OUTPUT_IMG_SIZE = (112,112,3)  # size of output images (square images)

    det_model = insightface.model_zoo.get_model(path_model)
    det_model.prepare(ctx_id=0, input_size=(640,640))

    start = datetime.datetime.now()


    cnt_noface = 0
    l_n_faces_detected = []
    l_det_scores = []
    l_a_img_aligned = []
    l = []

    for i, p in enumerate(df_dataset["path"].values):
        if (i%100==0) & (i>0):
            running_time = datetime.datetime.now() - start
            print(f"{i} photos downloaded\t{(i/len(df_dataset))*100:.3f} % done\tRunning time: {running_time.seconds}\tRemaining: {running_time.seconds/i*(len(df_dataset)-i):.0f}")
        img_arr = np.asarray(Image.open(p))
        bboxes, kpss = det_model.detect(img_arr)
        l.append([bboxes, kpss])
        n_faces_detected = len(bboxes)
        l_n_faces_detected.append(n_faces_detected)
        l_det_scores.append(bboxes[:,4])
        if n_faces_detected:
            bbox, det_score = bboxes[0, 0:4], bboxes[0, 4]
            kps = kpss[0]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            arr_img_aligned = insightface.utils.face_align.norm_crop(img_arr, landmark=face.kps, image_size=OUTPUT_IMG_SIZE[0])  # might have to adjust this for different input sizes
            l_a_img_aligned.append(arr_img_aligned)
        else:
            cnt_noface += 1
            l_a_img_aligned.append(np.zeros(OUTPUT_IMG_SIZE).astype("uint8"))
    df_dataset["retina_n_faces_detected"] = l_n_faces_detected
    a_aligned = np.stack(l_a_img_aligned)

    with open(os.path.join(f"{DATA_PATH}/{dataset_name}", "detection_retina.pkl"), 'wb') as handle:
        pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_dataset.to_parquet(os.path.join(f"{DATA_PATH}/{dataset_name}", "df_dataset.parquet"))
    np.savez_compressed(f"{DATA_PATH}/{dataset_name}/detection_retina_{'-'.join([str(x) for x in OUTPUT_IMG_SIZE])}.npz", a = a_aligned)


    ### sanity checks ###

    # number of images stays constant
    assert len(l_a_img_aligned) == len(df_dataset)
    # number of faces must be >= 0 for each image, and there is one detection_score for each face
    assert (df_dataset["retina_n_faces_detected"]>=0).all()
    # the number of images with no faces is controlled during inference with the counter (cnt_face) - 
    assert cnt_noface == (a_aligned.max(axis=(1,2,3)) == 0).sum() 
    assert cnt_noface == (df_dataset["retina_n_faces_detected"]==0).sum()
    # default data type for images
    assert a_aligned.dtype == "uint8"