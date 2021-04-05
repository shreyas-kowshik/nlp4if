import nlpaug.augmenter.word as naw
import nlpaug.flow as nafc
from nlpaug.util import Action
from random import randint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dtp", "--data_train_path", type=str, default="data/english/v1/v1/",
                    help="Expects a path to training folder")
parser.add_argument("-ddp", "--data_dev_path", type=str, default="data/english/v2/v2/",
                    help="Expects a path to dev folder")
args = parser.parse_args()

TRAIN_FILE=args.data_train_path+"covid19_disinfo_binary_english_train.tsv"
DEV_FILE=args.data_dev_path+"covid19_disinfo_binary_english_dev_input.tsv"

df_train = pd.read_csv(TRAIN_FILE, sep='\t')

aug_ins = naw.ContextualWordEmbsAug(
    model_path='roberta-large', action="insert")
aug_sub = naw.ContextualWordEmbsAug(
    model_path='roberta-large', action="substitute")

index = randint(0, len(df_train)-1)
text = df_train['tweet_text'][index]

print("Original:")
print(text)
print("Augmented (insert) Text:")
print(aug_ins.augment(text))
print("Augmented (subs) Text:")
print(aug_sub.augment(text))    
