import numpy as np
import os
import sys
import argparse
from utils.ensemble_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-ddp", "--data_dev_path", type=str, default="data/english/v3/v3/",
                    help="Expects a path to dev folder")
parser.add_argument("-dtp", "--data_test_path", type=str, default="data/english/test-input/test-input/",
                    help="Expects a path to training folder")
parser.add_argument("-psn", "--pred_save_name", type=str, default=None,
                    help="Name to save pred file with")
parser.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="Batch Size")
parser.add_argument("-wdbr", "--wandb_run", type=str, required=True,
                    help="Wandb Run Name")
args = parser.parse_args()

wandb.init(name=args.wandb_run, project='nlp_runs', entity='nlp4if')
TEST_FILE = args.data_test_path+"covid19_disinfo_binary_english_test_input.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

print('SET retrive_From_wandb paths')
retrive_from_wandb = [
    ["nlp4if/nlp_runs/2we87c77", "ensemble_aug_inverse_weights_roberta_small.pt"],
]

print('Using following file paths', retrive_from_wandb)
if len(retrive_from_wandb)==0:
    print("GIVE retrive_from_wandb paths")

for fp, fn in retrive_from_wandb:
    print(f'python get_wandb_files.py -fp {fp} -fn {fn}')
    os.system(f'python get_wandb_files.py -fp {fp} -fn {fn}')

MODEL_PATHS = [i[1] for i in retrive_from_wandb]

PRED_SAVE_NAME=args.pred_save_name
if PRED_SAVE_NAME is None:
    PRED_SAVE_NAME=('_').join([i.split('.')[0] for i in MODEL_PATHS])+".tsv"

print('File will be saved in test_preds/'+PRED_SAVE_NAME)
eval_ensemble_test(MODEL_PATHS, TEST_FILE, DEV_FILE, PRED_SAVE_NAME, device=torch.device('cuda'), use_glove_fasttext=False)
os.system(f'python format_checker/main.py -p {"test_preds/"+PRED_SAVE_NAME}')