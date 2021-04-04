import numpy as np
import pandas as pd
import random
import copy
import argparse
from distutils.util import strtobool
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, BertTokenizer
# Model Imports #
from models.bert_basic import *
from models.BertAttentionClasswise import *
# Utils Imports #
from utils.preprocess import *
from utils.train_utils import *
import wandb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

print("!!!!!!! NOTE: TASK NUMBERS ARE FROM 1-7")
#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="data/english/v1/v1/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="data/english/v2/v2/",
                    help="Expects a path to dev folder")
parser.add_argument("-model", "--model_to_use", type=str, default="bert_train_emb",
                    help="Which model to use")
parser.add_argument("-bbase", "--bert_base", type=str, default="bert-large-cased",
                    help="Which bert base model to use")
parser.add_argument("-task", "--task", type=int, required=True,
                    help="Maximum sequence length")
parser.add_argument("-msl", "--max_seq_len", type=int, default=56,
                    help="Maximum sequence length")
parser.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="Batch Size")
parser.add_argument("-e", "--epochs", type=int, required=True,
                    help="Epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5,
                    help="Learning Rate of non-embedding params")
parser.add_argument("-dprob", "--dropout_prob", type=float, default=0.1,
                    help="Dropout Probability")
parser.add_argument("-lr_emb", "--learning_rate_embeddings", type=float, default=2e-5,
                    help="Learning Rate of embedding params")
parser.add_argument("-device", "--device", type=str, default="cuda",
                    help="Device")
parser.add_argument("-loss", "--loss_type", type=str, default="single_class",
                    help="Loss")
parser.add_argument("-wdbr", "--wandb_run", type=str, required=True,
                    help="Wandb Run Name")
parser.add_argument("-log_to_wnb", "--log_to_wnb", type=strtobool, default=True,
                    help="Wandb Run Name")
parser.add_argument("-save_emb", "--save_emb", type=strtobool, default=False,
                    help="Save Model Embeddings")

args = parser.parse_args()

# Add initial values here #
if args.log_to_wnb==True:
    wandb.init(name=args.wandb_run, project='nlp_runs', entity='nlp4if')
    wandb.config.update(args)
###########################

### Base Parameters ###
device = torch.device(args.device)
print(device)
TRAIN_FILE=args.data_train_path+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"
#######################

if args.task in [2, 3, 4, 5]:
    num_classes=3
else:
    num_classes=2    

### Data Preparation ###
sentences, labels, train_les = process_data(TRAIN_FILE)
sentences_dev, labels_dev, train_les_dev = process_data(DEV_FILE)

if args.task in [2, 3, 4, 5]:
    indices_where_label_1_yes = np.where(labels_dev[:,0]==1)

labels=labels[:,args.task-1]
labels_dev=labels_dev[:,args.task-1]
labels_dev=labels_dev[indices_where_label_1_yes]
sentences_dev=sentences_dev[indices_where_label_1_yes]

train_x = sentences
train_y = labels
val_x = sentences_dev
val_y = labels_dev

# Generate class weights #
generate_class_weights(TRAIN_FILE)
########################

### Tokenize Data ###
tokens_train = bert_tokenize(train_x, args.max_seq_len, bert_base=args.bert_base)
tokens_val = bert_tokenize(val_x, args.max_seq_len, bert_base=args.bert_base)

#####################

### Dataloader Preparation ###
# convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_y.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_y.tolist())

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=args.batch_size)

################################

### Model Preparation ###
model = BERTAttentionSingleTask(num_classes, freeze_bert_params=True, bert_base=args.bert_base)
model = model.to(device)

if args.log_to_wnb==True:
    wandb.watch(model, log="all")
    
model = model.to(device)
#########################

### Train ###
model = train_v2_sc(model, train_dataloader, val_dataloader, args.device, args.epochs, 
           lr1=args.learning_rate, lr2=args.learning_rate_embeddings, loss_type=args.loss_type)

score = evaluate_model_sc(model, val_dataloader, args.device)
print(score)

### Print Stats ###
print("---Final Dev Stats---")
score = evaluate_model_sc(model, val_dataloader, args.device)
print(score)

# Save summary wandb
if args.log_to_wnb==True:
    wandb.run.summary['Validation F1-Score']: val_scores['f1']
    wandb.run.summary['Validation Precision']: val_scores['p_score']
    wandb.run.summary['Validation Accuracy']: val_scores['acc']
    wandb.run.summary['Validation Recall']: val_scores['r_score']

if args.save_emb==True:
	train_emb = get_model_embeddings(model, train_dataloader, args.device)
	val_emb = get_model_embeddings(model, val_dataloader, args.device)
	np.save(os.path.join(wandb.run.dir, "train_emb.npy"), train_emb)
	np.save(os.path.join(wandb.run.dir, "val_emb.npy"), val_emb)
