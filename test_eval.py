import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_test_path", type=str, default="data/english/v3/v3_augmented/",
                    help="Expects a path to training folder")
parser.add_argument("-model_name", "--model_to_use", type=str, required=True,
                    help="Which model to use")
args = parser.parse_args()


def build_test_loader(data_path):
    data = pd.read_csv(data_path, sep='\t')
    print("Dropping rows that contain nan in Q6_label or Q7_label")
    data=data.dropna(subset=['q7_label', 'q6_label'])  
    data["tweet_text"] = data["tweet_text"].apply(lambda x:unidecode(x))  
    sentences = data["tweet_text"]
    return np.array(sentences)