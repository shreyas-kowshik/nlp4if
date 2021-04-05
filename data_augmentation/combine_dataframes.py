import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="../data/english/v1/v1/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="../data/english/v2/v2/",
                    help="Expects a path to dev folder")
args = parser.parse_args()

TRAIN_FILE=args.data_train_path+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

df_train = pd.read_csv(TRAIN_FILE, sep='\t')
df_train_es=pd.read_csv('augmented_datasets/df_train_es.tsv', sep='\t')
df_train_fr=pd.read_csv('augmented_datasets/df_train_fr.tsv', sep='\t')
df_train_de=pd.read_csv('augmented_datasets/df_train_de.tsv', sep='\t')