'''
How to run:
Give trial name as arg using -wdbr
python grid_search.py -wdbr dry_run

If the grid search gets interputed during execution, then use start_from arg to give from which index it has to continue
ex: Say 1-30 run is complete. Then use "python grid_search.py -wdbr dry_run -start_from 31" to continue grid search

Make to set model_to_use arg in os.system call if default model is not being used
'''

import os 
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-wdbr", "--wandb_run", type=str, required=True,
                    help="Wandb Run Name")
parser.add_argument("-start_from", "--start_from", type=int, default=0,
                    help="If grid search is interrupted, specify which run number to continue from")

args = parser.parse_args()

WANDB_TRIAL_NAME = args.wandb_run

batch_size=[16, 32]
learning_rate = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
learning_rate_embeddings = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

i=0
for bs in batch_size:
    for lr in learning_rate:
        for lr_emb in learning_rate_embeddings:
            i+=1
            if i>=args.start_from:
                WANDB_TRIAL_NAME = ('_').join([args.wandb_run, str(i)])
                print(f'python bert_train.py -bs {bs} -lr {lr} -lr_emb {lr_emb} -e 100 -wdbr {WANDB_TRIAL_NAME}')
                os.system(f'python bert_train.py -bs {bs} -lr {lr} -lr_emb {lr_emb} -e 100 -wdbr {WANDB_TRIAL_NAME}')