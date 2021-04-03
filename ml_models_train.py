import numpy as np
import pandas as pd
import random
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import argparse
from distutils.util import strtobool
from utils.preprocess import *
from utils.train_utils import *
import os
import wandb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="data/english/v1/v1/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="data/english/v2/v2/",
                    help="Expects a path to dev folder")
parser.add_argument("-model", "--model_to_use", type=str, default="RandomForest",
                    help="Which model to use")
parser.add_argument("-wdbr", "--wandb_run", type=str, 
                    help="Wandb Run Name")
parser.add_argument("-log_to_wnb", "--log_to_wnb", type=strtobool, default=True,
                    help="Wandb Run Name")
parser.add_argument("-use_emb", "--use_emb", type=strtobool, default=True,
                    help="Whether to use embeddings")
parser.add_argument("-embf", "--emb_folder", type=str, default="bert_attn_cw_prepro_head7"
                    help="Folder Name where Embedding is stored")


args = parser.parse_args()

# Set data paths
TRAIN_FILE=args.data_train_path+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

# Process and clean data 
df_train = pd.read_csv(TRAIN_FILE, sep='\t')
df_val = pd.read_csv(DEV_FILE, sep='\t')
df_train=df_train.dropna(subset=['q7_label', 'q6_label'])  
df_val=df_val.dropna(subset=['q7_label', 'q6_label'])
df_train = preprocess_cleaning(copy.deepcopy(df_train))
df_val = preprocess_cleaning(copy.deepcopy(df_val))
special_features = ['num_url', 'num_user_id', 'num_emoji', 'has_url', 'has_emoji', 'num_hashtags', 'num_user_mention', 'num_punctuation']

# Prepare labels, class weights
label_list = ['q1_label', 'q2_label', 'q3_label', 'q4_label', 'q5_label', 'q6_label', 'q7_label']
cw = generate_class_weights(TRAIN_FILE, return_weights=True)
Y_TRAIN_FULL = convert_label(df_train[label_list].fillna('nan').to_numpy())
Y_VAL_FULL = convert_label(df_val[label_list].fillna('nan').to_numpy())



# Generate word and char tfidf ngrams
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',   
    ngram_range=(1, 2),
    max_features=5000)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(2, 3),
    max_features=5000)

char_vectorizer.fit(df_train['text_cleaned'])
X_char_train = char_vectorizer.transform(df_train['text_cleaned']).toarray()
X_char_val = char_vectorizer.transform(df_val['text_cleaned']).toarray()

word_vectorizer.fit(df_train['text_cleaned'])
X_word_train = word_vectorizer.transform(df_train['text_cleaned']).toarray()
X_word_val = word_vectorizer.transform(df_val['text_cleaned']).toarray()

# Stack word tfidf, char tfidf, hand-crafted features
X_train = np.hstack([X_char_train, X_word_train, df_train[special_features].to_numpy()])
X_val = np.hstack([X_char_val, X_word_val, df_val[special_features].to_numpy()])

# Embedding features
if args.use_emb:
    train_emb = np.load(os.path.join('data/', args.emb_folder, 'train_emb.npy'))
    val_emb = np.load(os.path.join('data/', args.emb_folder, 'val_emb.npy'))
    X_train = np.hstack([X_train, train_emb])
    X_val = np.hstack([X_val, val_emb])
    print("Embeddings stacked!")


if args.model_to_use=='RandomForest':
    y_pred_train, y_pred_val = [], []
    for i in range(7):
        Y_train = Y_TRAIN_FULL[:,i].astype('int')
        clf = RandomForestClassifier(class_weight={n:j for n, j in enumerate(cw[i])}, random_state=42)
        clf.fit(X_train, Y_train)
        y_pred_train.append(clf.predict(X_train))
        y_pred_val.append(clf.predict(X_val))
    
    y_pred_train=np.vstack(y_pred_train).T
    y_pred_val=np.vstack(y_pred_val).T

    print('----------------------- TRAIN SCORES -----------------------')
    train_scores = evaluate_model_ml(y_pred_train.astype(np.int64), y_pred_train.astype(np.int64))
    display_metrics(train_scores)
    print('----------------------- VAL SCORES -----------------------')
    scores, val_preds, val_gt = evaluate_model_ml(y_pred_val.astype(np.int64), Y_VAL_FULL.astype(np.int64), return_files=True)
    display_metrics(scores)
