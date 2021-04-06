import numpy as np
import os
import sys
import argparse
from utils.ensemble_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, required=True,
                    help="Epochs")
parser.add_argument("-wdbr", "--wandb_run", type=str, required=True,
                    help="Wandb Run Name")

args = parser.parse_args()

EPOCHS=args.epochs
DEV_FILE='data/english/v3/v3/covid19_disinfo_binary_english_dev_input.tsv'

# Train BERT
WANDB_TRIAL_NAME=args.wandb_run+'_bert_large'
print("Training BERT Large")
print(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS} -wdbr {WANDB_TRIAL_NAME} -model bert_attn --base bert-large-cased --save_model True')
os.system(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model bert_attn --base bert-large-cased --save_model True')

WANDB_TRIAL_NAME=args.wandb_run+'_bert_small'
print("Training BERT Small")
print(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model bert_attn_classwise --base bert-base-uncased --save_model True')
os.system(f'python bert_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model bert_attn_classwise --base bert-base-uncased --save_model True')

# Train Roberta
WANDB_TRIAL_NAME=args.wandb_run+'_roberta_large'
print("Training RoBERTa Large")
print(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn --base roberta-large --save_model True')
os.system(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn --base roberta-large --save_model True')

WANDB_TRIAL_NAME=args.wandb_run+'_roberta_small'
print("Training RoBERTa Small")
print(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn_classwise --base roberta-base --save_model True')
os.system(f'python roberta_train.py -bs 32 -lr 5e-5 -lr_emb 5e-6 -e {EPOCHS}  -wdbr {WANDB_TRIAL_NAME} -model roberta_attn_classwise --base roberta-base --save_model True')


# Save results
scores = eval_ensemble(args.wandb_run, DEV_FILE)
display_metrics(scores)

# Save summary wandb
wandb.init(name=args.wandb_run, project='nlp_runs', entity='nlp4if')
wandb.config.update(args)
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
wandb.run.summary['Validation Q1 Precision'] = scores['p_score'][0]
wandb.run.summary['Validation Q2 Precision'] = scores['p_score'][1]
wandb.run.summary['Validation Q3 Precision'] = scores['p_score'][2]
wandb.run.summary['Validation Q4 Precision'] = scores['p_score'][3]
wandb.run.summary['Validation Q5 Precision'] = scores['p_score'][4]
wandb.run.summary['Validation Q6 Precision'] = scores['p_score'][5]
wandb.run.summary['Validation Q7 Precision'] = scores['p_score'][6]
