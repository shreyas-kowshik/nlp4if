## WIP

import os 
import sys

batch_size=[16, 32]
learning_rate = [1e-5, 3e-5, 1e-4]
learning_rate_embeddings = [1e-5, 2e-5, 3e-5]

for lr in learning_rate:
    for lr_emb in learning_rate_embeddings:
        print(f'python bert_train.py -lr {lr} -lr_emb {lr_emb}')
        # os.system(f'python bert_train.py -lr {lr} -lr_emb {lr_emb}')