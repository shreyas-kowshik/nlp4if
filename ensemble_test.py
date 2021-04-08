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
parser.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="Batch Size")
parser.add_argument("-wdbr", "--wandb_run", type=str, required=True,
                    help="Wandb Run Name")
args = parser.parse_args()

wandb.init(name=args.wandb_run, project='nlp_runs', entity='nlp4if')
TEST_FILE = args.data_test_path+"covid19_disinfo_binary_english_test_input.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

MODEL_PATHS = ['ensemble_aug_inverse_weights_roberta_small.pt']
if len(MODEL_PATHS)==0:
    print("GIVE MODEL PATHS")

eval_ensemble_test(MODEL_PATHS, TEST_FILE, DEV_FILE, device=torch.device('cuda'), use_glove_fasttext=False)