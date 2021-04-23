import wandb
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--file_path", type=str, required=True,
                    help="Expects a path to wnb model")
parser.add_argument("-fn", "--file_name", type=str, required=True,
                    help="Expects name of file to dowload")                    
args = parser.parse_args()

if not os.path.isfile(args.file_name):
    api = wandb.Api()
    run = api.run(args.file_path)
    run.file(args.file_name).download()
else:
    print('File already exists')