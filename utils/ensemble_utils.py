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

def train_bert(nmodel, training_dataloader, val_dataloader, device, epochs = 4, lr1=2e-5, lr2=1e-4, loss_type="classwise_sum", model_name="bert_base_uncase"):
    total_steps = len(training_dataloader) * epochs
    bert_params = nmodel.embeddings
    bert_named_params = ['embeddings.'+name_ for name_, param_ in bert_params.named_parameters()]
    model_named_params = [name_ for name_, param_ in nmodel.named_parameters()]
    other_named_params = [i for i in model_named_params if i not in bert_named_params]
    params = []

    for name, param in nmodel.named_parameters():
        if name in other_named_params:
            params.append(param)
    
    optimizer1 = AdamW(bert_params.parameters(), lr=lr2, eps = 1e-8)
    optimizer2 = AdamW(params, lr=lr1, eps = 1e-8)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    loss_fn = None
    if loss_type == "classwise_sum":
        loss_fn = classwise_sum
    elif loss_type == "single_class":
        loss_fn = nn.CrossEntropyLoss()
    else:
        print("Loss {} Not Defined".format(loss_type))
        sys.exit(1)

    # Load class weights for loss function
    wts = []
    for i in range(7):
        wts.append(torch.Tensor(np.load('data/class_weights/q' + str(i+1) + '.npy')).to(device))

    best_model = copy.deepcopy(nmodel)
    best_f1 = 0
    for epoch_i in tqdm(range(0, epochs)):
        total_train_loss = 0
        nmodel.train()
        for step, batch in enumerate(training_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch

            ypreds = nmodel(sent_id, mask)
            if loss_type == "classwise_sum":
                loss = loss_fn(ypreds, labels, wts)
            elif loss_type == "single_class":
                loss = loss_fn(ypreds, labels)

            total_train_loss += loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_params.parameters(), 1.0)
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

        print('Total Train Loss =', total_train_loss)
        print('#############    Validation Set Stats')
        train_scores = evaluate_model(nmodel, training_dataloader, device)
        scores, val_preds, val_gt = evaluate_model(nmodel, val_dataloader, device, return_files=True)
        display_metrics(scores)

        # Log scores on wandb
        wandb.log({
            # Mean Scores
            model_name + ' Training loss': total_train_loss,
            model_name + ' Training Mean F1-Score': np.mean(train_scores['f1']),
            model_name + ' Training Accuracy': np.mean(train_scores['acc']),
            model_name + ' Training Mean Precision': np.mean(train_scores['p_score']),
            model_name + ' Training Mean Recall': np.mean(train_scores['r_score']),
            model_name + ' Validation Mean F1-Score': np.mean(scores['f1']),
            model_name + ' Validation Accuracy': np.mean(scores['acc']),
            model_name + ' Validation Mean Precision': np.mean(scores['p_score']),
            model_name + ' Validation Mean Recall': np.mean(scores['r_score']),
            model_name + ' Training Q1 F1 Score':train_scores['f1'][0],
            model_name + ' Training Q2 F1 Score':train_scores['f1'][1],
            model_name + ' Training Q3 F1 Score':train_scores['f1'][2],
            model_name + ' Training Q4 F1 Score':train_scores['f1'][3],
            model_name + ' Training Q5 F1 Score':train_scores['f1'][4],
            model_name + ' Training Q6 F1 Score':train_scores['f1'][5],
            model_name + ' Training Q7 F1 Score':train_scores['f1'][6],
            model_name + ' Validation Q1 F1 Score':scores['f1'][0],
            model_name + ' Validation Q2 F1 Score':scores['f1'][1],
            model_name + ' Validation Q3 F1 Score':scores['f1'][2],
            model_name + ' Validation Q4 F1 Score':scores['f1'][3],
            model_name + ' Validation Q5 F1 Score':scores['f1'][4],
            model_name + ' Validation Q6 F1 Score':scores['f1'][5],
            model_name + ' Validation Q7 F1 Score':scores['f1'][6]
        }, step=epoch_i)

        if np.mean(scores['f1']) > best_f1:
            best_model = copy.deepcopy(nmodel)
            best_f1 = np.mean(scores['f1'])

            # Save model to wandb
            # wandb.save('checkpoints_best_val.pt')
            # nmodel.save(os.path.join(wandb.run.dir, "best_val.pt"))
            # torch.save(nmodel.state_dict(), os.path.join(wandb.run.dir, "best_val.pth"))

            np.savetxt(os.path.join(wandb.run.dir, "best_val_preds.tsv"), val_preds, delimiter="\t",fmt='%s')
            np.savetxt(os.path.join(wandb.run.dir, "best_val_grount_truth.tsv"), val_gt, delimiter="\t",fmt='%s')


    
    return best_model
