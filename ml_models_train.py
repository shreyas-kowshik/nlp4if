import numpy as np
import pandas as pd
import random
import copy
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import argparse
from distutils.util import strtobool
from utils.preprocess import *
from utils.train_utils import *
import wandb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="data/english/v3/v3/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="data/english/v3/v3/",
                    help="Expects a path to dev folder")
parser.add_argument("-model", "--model_to_use", type=str, default="RandomForest",
                    help="Which model to use")
parser.add_argument("-max_features_ngram", "--max_features_ngram", type=int, default=5000,
                    help="Which model to use")                    
parser.add_argument("-wdbr", "--wandb_run", type=str, required=True,
                    help="Wandb Run Name")
parser.add_argument("-log_to_wnb", "--log_to_wnb", type=strtobool, default=True,
                    help="Wandb Run Name")
parser.add_argument("-use_emb", "--use_emb", type=strtobool, default=True,
                    help="Whether to use embeddings")
parser.add_argument("-embf", "--emb_folder", type=str, default="bert_attn_cw_prepro_head7",
                    help="Folder Name where Embedding is stored")
args = parser.parse_args()

# Add initial values here #
if args.log_to_wnb==True:
    wandb.init(name=args.wandb_run, project='nlp_runs', entity='nlp4if')
    wandb.config.update(args)

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

X_word_train, X_word_val, X_char_train, X_char_val = build_tfidf_ngrams(df_train, df_val, max_features=args.max_features_ngram)

# Stack word tfidf, char tfidf, hand-crafted features
X_train = np.hstack([X_char_train, X_word_train, df_train[special_features].to_numpy()])
X_val = np.hstack([X_char_val, X_word_val, df_val[special_features].to_numpy()])

# Embedding features
if args.use_emb:
    train_emb = np.load(os.path.join('data/embeddings', args.emb_folder, 'train_emb.npy'))
    val_emb = np.load(os.path.join('data/embeddings', args.emb_folder, 'val_emb.npy'))
    X_train = np.hstack([X_train, train_emb])
    X_val = np.hstack([X_val, val_emb])
    print("Embeddings stacked!")

model_dict = {'RandomForest':RandomForestClassifier, 'XGBoost':XGBClassifier, 'LogisticRegression':LogisticRegression}
clf, y_pred_train, y_pred_val = train_ml(model_dict[args.model_to_use],  X_train, Y_TRAIN_FULL, X_val, cw)

print('----------------------- TRAIN SCORES -----------------------')
train_scores = evaluate_model_ml(y_pred_train.astype(np.int64), y_pred_train.astype(np.int64))
display_metrics(train_scores)
print('----------------------- VAL SCORES -----------------------')
scores, val_preds, val_gt = evaluate_model_ml(y_pred_val.astype(np.int64), Y_VAL_FULL.astype(np.int64), return_files=True)
display_metrics(scores)

# Save summary wandb
if args.log_to_wnb==True:
    wandb.run.summary['Validation Mean F1-Score'] = np.mean(scores['f1'])
    wandb.run.summary['Validation Accuracy'] = np.mean(scores['acc'])
    wandb.run.summary['Validation Mean Precision'] = np.mean(scores['p_score'])
    wandb.run.summary['Validation Mean Recall'] = np.mean(scores['r_score'])
    wandb.run.summary['Validation Q1 F1 Score'] = scores['f1'][0]
    wandb.run.summary['Validation Q2 F1 Score'] = scores['f1'][1]
    wandb.run.summary['Validation Q3 F1 Score'] = scores['f1'][2]
    wandb.run.summary['Validation Q4 F1 Score'] = scores['f1'][3]
    wandb.run.summary['Validation Q5 F1 Score'] = scores['f1'][4]
    wandb.run.summary['Validation Q6 F1 Score'] = scores['f1'][5]
    wandb.run.summary['Validation Q7 F1 Score'] = scores['f1'][6]
    wandb.run.summary['Validation Q1 F1 Precision'] = scores['p_score'][0]
    wandb.run.summary['Validation Q2 F1 Precision'] = scores['p_score'][1]
    wandb.run.summary['Validation Q3 F1 Precision'] = scores['p_score'][2]
    wandb.run.summary['Validation Q4 F1 Precision'] = scores['p_score'][3]
    wandb.run.summary['Validation Q5 F1 Precision'] = scores['p_score'][4]
    wandb.run.summary['Validation Q6 F1 Precision'] = scores['p_score'][5]
    wandb.run.summary['Validation Q7 F1 Precision'] = scores['p_score'][6]

    wandb.run.summary['Train Mean F1-Score'] = np.mean(train_scores['f1'])
    wandb.run.summary['Train Accuracy'] = np.mean(train_scores['acc'])
    wandb.run.summary['Train Mean Precision'] = np.mean(train_scores['p_score'])
    wandb.run.summary['Train Mean Recall'] = np.mean(train_scores['r_score'])
    wandb.run.summary['Train Q1 F1 Score'] = train_scores['f1'][0]
    wandb.run.summary['Train Q2 F1 Score'] = train_scores['f1'][1]
    wandb.run.summary['Train Q3 F1 Score'] = train_scores['f1'][2]
    wandb.run.summary['Train Q4 F1 Score'] = train_scores['f1'][3]
    wandb.run.summary['Train Q5 F1 Score'] = train_scores['f1'][4]
    wandb.run.summary['Train Q6 F1 Score'] = train_scores['f1'][5]
    wandb.run.summary['Train Q7 F1 Score'] = train_scores['f1'][6]
    wandb.run.summary['Train Q1 F1 Precision'] = train_scores['p_score'][0]
    wandb.run.summary['Train Q2 F1 Precision'] = train_scores['p_score'][1]
    wandb.run.summary['Train Q3 F1 Precision'] = train_scores['p_score'][2]
    wandb.run.summary['Train Q4 F1 Precision'] = train_scores['p_score'][3]
    wandb.run.summary['Train Q5 F1 Precision'] = train_scores['p_score'][4]
    wandb.run.summary['Train Q6 F1 Precision'] = train_scores['p_score'][5]
    wandb.run.summary['Train Q7 F1 Precision'] = train_scores['p_score'][6]
