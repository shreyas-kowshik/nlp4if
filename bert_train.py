import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, BertTokenizer
# Config Import #
from config.config import *
# Model Imports #
from models.bert_basic import *
# Utils Imports #
from utils.preprocess import *
from utils.train_utils import *

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

### Base Parameters ###
device = torch.device(device_name)
print(device)
TRAIN_FILE=DATA_PATH+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=DATA_PATH_DEV+"covid19_disinfo_binary_english_dev_input.tsv"
#######################

### Data Preparation ###
sentences, labels, train_les = process_data(TRAIN_FILE)
sentences_dev, labels_dev, train_les_dev = process_data(DEV_FILE)

idx = [i for i in range(len(sentences))]
np.random.shuffle(idx)
train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
train_x = sentences[train_idx]
val_x = sentences[val_idx]
train_y = labels[train_idx, :]
val_y = labels[val_idx, :]
########################

### Tokenize Data ###
tokens_train = bert_tokenize(train_x, MAX_SEQ_LEN)
tokens_val = bert_tokenize(val_x, MAX_SEQ_LEN)
tokens_dev = bert_tokenize(sentences_dev, MAX_SEQ_LEN)

#####################

### Dataloader Preparation ###
# convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_y.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_y.tolist())

dev_seq = torch.tensor(tokens_dev['input_ids'])
dev_mask = torch.tensor(tokens_dev['attention_mask'])
dev_y = torch.tensor(labels_dev.tolist())

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=BATCH_SIZE)

# dev tensors
dev_data = TensorDataset(dev_seq, dev_mask, dev_y)
# sampler for sampling the data during training
dev_sampler = SequentialSampler(dev_data)
# dataLoader for validation set
dev_dataloader = DataLoader(dev_data, sampler = dev_sampler, batch_size=BATCH_SIZE)

################################

### Model Preparation ###
model = BERTBasic(freeze_bert_params=False)
model = model.to(device)
#########################

### Train ###
model = train_v2(model, train_dataloader, val_dataloader, device, EPOCHS, learning_rate)

classwise_acc, total_acc = evaluate(model, dev_dataloader, device)
print('Classwise accuracy: ', classwise_acc, '\nTotal accuracy: ', total_acc)