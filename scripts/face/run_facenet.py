import argparse, datetime
import numpy as np
from keras_facenet import FaceNet

DATA_PATH = "../../data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is the description"
    )
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--path_cropped_imgs", required=True, type=str)
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    path_cropped_imgs = args.path_cropped_imgs
    
    a_aligned = np.load(path_cropped_imgs)["a"]
    model = FaceNet()
    print("Instantiated FaceNet model")
    
    start = datetime.datetime.now()

    chunsize = 1000
    l_embeddings = []
    for i in range(len(a_aligned)//chunsize):
        running_time = datetime.datetime.now() - start
        print(f"{i*chunsize} photos embedded\tRunning time: {running_time.seconds}")
        l_embeddings.append(model.embeddings(a_aligned[i*chunsize:(i+1)*chunsize]))
    l_embeddings.append(model.embeddings(a_aligned[(i+1)*chunsize:]))
    a_embeddings = np.concatenate(l_embeddings)
    np.savez_compressed(f"{DATA_PATH}/embeddings/{dataset_name}/embeddings_retina_facenet.npz", a = a_embeddings)