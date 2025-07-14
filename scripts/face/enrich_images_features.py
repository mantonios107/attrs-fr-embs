## This scripts computes features of images
## input: df_dataset.parquet file, path/to/images in .npz format with images stored in "a" - np.load("../data/path/file.npz")["a"]

import argparse
import cv2
import pandas as pd
import numpy as np

def compute_imageStatisticsBasic(a_aligned):
    l_column_names = ["rgb_r", "rgb_g", "rgb_b", "hsv_h", "hsv_s", "hsv_v"]
    l = []
    for img in a_aligned:
        rgb_r = np.mean(img[:,:, 0])
        rgb_g = np.mean(img[:,:, 1])
        rgb_b = np.mean(img[:,:, 2])
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_h = np.mean(img_hsv[:,:, 0])
        hsv_s = np.mean(img_hsv[:,:, 1])
        hsv_v = np.mean(img_hsv[:,:, 2])
        l.append([rgb_r, rgb_g, rgb_b, hsv_h, hsv_s, hsv_v])
    return pd.DataFrame(l, columns=l_column_names)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="computes summary statistics of images")
    parser.add_argument("--df_parquet", type=str, required=True)
    parser.add_argument("--img_file_path", type=str, required=True)
    args = parser.parse_args()
    df_dataset = pd.read_parquet(args.df_parquet)
    a_aligned = np.load(args.img_file_path)["a"]
    assert len(df_dataset)==len(a_aligned)  # sanity check
    print("Loaded df_parquet and img_file_path")
    
    df_imageStatistics = compute_imageStatisticsBasic(a_aligned)
    df_dataset = pd.concat([df_dataset.drop(columns=df_imageStatistics.columns[np.isin(df_imageStatistics.columns, df_dataset.columns)])
                       ,df_imageStatistics]
                       ,axis=1)
    
    df_dataset.to_parquet(args.df_parquet)