import numpy as np
import copy
from tqdm import tqdm
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.losses import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support
import wandb
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scorer.main import *
from utils.preprocess import *
from utils.train_utils import *

def bert_soft_preds(nmodel, test_dataloader, device):
    nmodel.eval()
    y_preds = []
    y_test = []
    init=True

    for i, batch in enumerate(test_dataloader):
        batch = [r.to(device) for r in batch]
        inc_feats = None
        sent_id, mask, labels = batch

        with torch.no_grad():
            ypreds = nmodel(sent_id, mask).cpu().numpy() # 7 outputs here
            for i, class_preds in enumerate(ypreds):
                if init:
                    y_preds.append(class_preds)
                else:
                    y_preds[i] = np.vstack([y_preds[i], class_preds])
            init=False

            labels=labels.cpu().numpy()
            y_preds.append(ypred)
            y_test.append(labels)

    y_test = np.vstack(y_test)
    return y_preds, y_test # y_preds : 7 elements, with each (N,num_classes) for each question, y_test is (N,7)

def get_dataloader_bert_type(dev_file, model_type, model_base, max_seq_len=56, batch_size=32):
    DEV_FILE=dev_file
    sentences_dev, labels_dev, train_les_dev = process_data(DEV_FILE)
    val_x = sentences_dev
    val_y = labels_dev

    ### Tokenize Data ###
    if model_type=="bert":
        tokens_dev = bert_tokenize(sentences_dev, max_seq_len, model_base)
    else:
        tokens_dev = roberta_tokenize(sentences_dev, max_seq_len, model_base)
    #####################

    ### Dataloader Preparation ###
    dev_seq = torch.tensor(tokens_dev['input_ids'])
    dev_mask = torch.tensor(tokens_dev['attention_mask'])
    dev_y = torch.tensor(labels_dev.tolist())

    dev_data = TensorDataset(dev_seq, dev_mask, dev_y)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler = dev_sampler, batch_size)

    return dev_dataloader


def eval_ensemble(wdbr, dev_file, device=torch.device('cuda'), use_glove_fasttext=False):
    model_soft_preds = []
    y_test = None
    for model_pth in os.listdir('bin'):
        if not wdbr in model_pth:
            continue

        model = torch.load(os.path.join('bin', model_pth)).to(device)
        val_dataloader=None
        if 'roberta' in model_pth:
            if 'large' in model_pth:
                val_dataloader = get_dataloader_bert_type(dev_file, 'roberta', 'roberta-large')
            else:
                val_dataloader = get_dataloader_bert_type(dev_file, 'roberta', 'roberta-base')
        elif 'bert' in model_pth:
            if 'large' in model_pth:
                val_dataloader = get_dataloader_bert_type(dev_file, 'bert', 'bert-large-cased')
            else:
                val_dataloader = get_dataloader_bert_type(dev_file, 'bert', 'bert-base-uncased')
        else:
            if not use_glove_fasttext:
                print("Error, model not understood in evluation")
                exit(1)

        ypreds, ytest = bert_soft_preds(model, val_dataloader, device)
        model_soft_preds.append(ypreds)

    for i in range(7):
        for j in range(1, len(model_soft_preds)):
            model_soft_preds[0][i] += model_soft_preds[j][i]

    model_soft_preds = model_soft_preds[0]
    for i in model_soft_preds:
        model_soft_preds[i] = np.argmax(model_soft_preds[i], axis=1, keepdims=True)
    y_preds = np.hstack(model_soft_preds)

    print(y_preds.shape)
    print(y_test.shape)

    y_preds = inverse_transform(y_preds)
    y_test = inverse_transform(y_test)

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    np.savetxt("tmp/preds_tem.tsv", y_preds, delimiter="\t",fmt='%s')
    np.savetxt("tmp/gt_tem.tsv", y_test, delimiter="\t",fmt='%s')

    truths, submitted = read_gold_and_pred('tmp/gt_tem.tsv', 'tmp/preds_tem.tsv')

    scores = {
        'acc': [],
        'f1': [],
        'p_score': [],
        'r_score': [],
    }
    all_classes = ["yes", "no"]
    for i in range(7):
        acc, f1, p_score, r_score = evaluate(truths[i+1], submitted[i+1], all_classes)
        for metric in scores:
            scores[metric].append(eval(metric))

    return scores
