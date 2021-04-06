import numpy as np
import pandas as pd 
import argparse
from utils.losses import *
import pandas as pd
import torchtext
import random
import os
from torchtext.data import TabularDataset
from torchtext import data
from sklearn.metrics import roc_auc_score,accuracy_score
import torch.nn as nn
import torch
import torch.optim as optim
import time
import spacy
import torch.nn.functional as F
from utils.glove_fastext_utils import *
from utils.train_utils import *

#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="data/english/v3/v3/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="data/english/v3/v3/",
                    help="Expects a path to dev folder")
parser.add_argument("-device", "--device", type=str, default="cuda",
                    help="Device")
parser.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="Batch Size")
parser.add_argument("-e", "--epochs", type=int, required=True,
                    help="Epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
                    help="Learning Rate of non-embedding params")
parser.add_argument("-wd", "--weight_decay", type=int, default=0,
                    help="Maximum sequence length")
parser.add_argument("-msl", "--max_seq_len", type=int, default=56,
                    help="Maximum sequence length")
args = parser.parse_args()

### Base Parameters ###
device = torch.device(args.device)
print(device)

TRAIN_FILE=args.data_train_path+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

# hyperparams
TEXT_LENGTH = args.max_seq_len
EMBEDDING_SIZE = 300
BATCH_SIZE = args.batch_size
VOCAB_SIZE=20000

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
nlp = spacy.load("en")
def tokenizer(text):
    filtered = ''.join([c if c not in filters else '' for c in text])
    return [token.text for token in nlp.tokenizer(filtered) if not token.is_space]

TEXT = data.Field(lower=True, batch_first=True,fix_length=TEXT_LENGTH, preprocessing=None, tokenize=tokenizer)
LABEL = data.Field(sequential=False,is_target=True, use_vocab=False, pad_token=None, unk_token=None)

convert_dataframe(TRAIN_FILE, 'train.tsv')
convert_dataframe(DEV_FILE, 'dev.tsv')

datafields = [('tweet_no', None),
              ('tweet_text', TEXT), 
              ("q1_label", LABEL), 
              ("q2_label", LABEL),
              ('q3_label', LABEL), 
              ('q4_label', LABEL),
              ('q5_label', LABEL),
              ('q6_label', LABEL),
              ('q7_label', LABEL),
              ]

train = TabularDataset(
    path='data_glove/train.tsv',
    format='tsv',
    skip_header=True,
    fields=datafields)

dev = TabularDataset(
    path='data_glove/dev.tsv',
    format='tsv',
    skip_header=True,
    fields=datafields)

# TEXT.build_vocab(train, max_size=20000, min_freq=5, vectors=['glove.840B.300d', 'fasttext.en.300d'])
TEXT.build_vocab(train, max_size=20000, min_freq=5, vectors='glove.840B.300d')

random.seed(1234)
train_iterator, valid_iterator = data.BucketIterator.splits((train, dev),
                                                            batch_size=BATCH_SIZE,
                                                            device=device,
                                                            shuffle=True,
                                                            sort_key=lambda x: len(x.tweet_text))


OUTPUT_DIM = 7
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = GloveNet(len(TEXT.vocab), EMBEDDING_SIZE, OUTPUT_DIM, PAD_IDX, TEXT.vocab.vectors,TEXT_LENGTH, 150).to(device)

wts = []
for i in range(7):
    wts.append(torch.Tensor(np.load('data/class_weights/q' + str(i+1) + '.npy')).to(device))

model=train_n_epochs(model, train_iterator, valid_iterator,  args.epochs,args.learning_rate,args.weight_decay, wts, device)
scores=evaluate_glove(train_iterator, model, device)
print(scores)
print(np.mean(scores['f1']))