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
parser.add_argument("-device", "--device", type=str, default="cuda",
                    help="Device")
args = parser.parse_args()

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

ckpt = torch.load(MODEL_PATHS[0])
val_dataloader = get_dataloader_bert_type(DEV_FILE, 'roberta', 'roberta-base')
device = torch.device(args.device)

model = ROBERTaAttentionClasswise(freeze_bert_params=False, base='roberta-base')
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()

for i, batch in enumerate(val_dataloader):
    batch = [r.to(device) for r in batch]
    sent_id, mask, labels = batch

    with torch.no_grad():
        attn_wts = model(sent_id, mask, get_attn_wt=True) # 7 outputs here

    attn_wts = [a.cpu().numpy() for a in attn_wts]
    print(attn_wts.shape)



