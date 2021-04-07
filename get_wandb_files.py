import wandb
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--file_path", type=str, required=True,
                    help="Expects a path to wnb model")
parser.add_argument("-fn", "--file_name", type=str, required=True,
                    help="Expects name of file to dowload")                    
args = parser.parse_args()

'''
ex: args.fp = "nlp4if/nlp_runs/2we87c77", args.fn="ensemble_aug_inverse_weights_roberta_small.pt"
'''

if not os.path.isfile(args.file_name):
    api = wandb.Api()
    run = api.run(args.mp)
    run.file(args.file_name).download()
else:
    print('File already exists')