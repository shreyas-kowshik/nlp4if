import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, required=True,
                    help="Epochs")
parser.add_argument("-wdbr", "--wandb_run", type=str, required=True,
                    help="Wandb Run Name")

EPOCHS=args.epochs
# Train BERT
WANDB_TRIAL_NAME=args.wandb_run+'_bert_large'
print("Training BERT Large")
print(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS} 100 -wdbr {WANDB_TRIAL_NAME} -model bert_attn --bert_base bert-large-cased --save_model True')
os.system(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model bert_attn --bert_base bert-large-cased --save_model True')

WANDB_TRIAL_NAME=args.wandb_run+'_bert_small'
print("Training BERT Small")
print(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model bert_attn_classwise --bert_base bert-base-uncased --save_model True')
os.system(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model bert_attn_classwise --bert_base bert-base-uncased --save_model True')

# Train Roberta
WANDB_TRIAL_NAME=args.wandb_run+'_roberta_large'
print("Training RoBERTa Large")
print(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn --bert_base roberta-large --save_model True')
os.system(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn --bert_base roberta-large --save_model True')

WANDB_TRIAL_NAME=args.wandb_run+'_roberta_small'
print("Training RoBERTa Small")
print(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn_classwise --bert_base roberta-base --save_model True')
os.system(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn_classwise --bert_base roberta-base --save_model True')

