import numpy as np
import copy
from tqdm import tqdm
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.losses import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support
import wandb
import os
from scorer.main import *

# Evaluate
def predict_labels(nmodel, test_dataloader, device):
    nmodel.eval()
    y_preds = []
    y_test = []
    for i, batch in enumerate(test_dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():        
            ypred = nmodel(sent_id, mask)
            ypred = np.vstack([np.argmax(i.cpu().numpy(), axis=1) for i in ypred]).T
            labels=labels.cpu().numpy()
            y_preds.append(ypred)
            y_test.append(labels)
    y_preds = np.vstack(y_preds)
    y_test = np.vstack(y_test)
    return y_preds, y_test


# def scorer_indiv(truths, submitted, all_classes):
#     print(np.unique(truths))
#     print(np.unique(submitted))
#     print("------------------")
#     acc = accuracy_score(truths, submitted)
#     f1 = f1_score(truths, submitted, labels=all_classes, average='weighted')
#     p_score = precision_score(truths, submitted, labels=all_classes, average='weighted')
#     r_score = recall_score(truths, submitted, labels=all_classes, average='weighted')
#     return acc, f1, p_score, r_score

# def scorer(y_true, y_pred, les):
#     """
#     les : label_encoders
#     """
#     truths = {}
#     truths_ids = {}
#     preds = {}

#     for i in range(7):
#         truths[i] = []
#         preds[i] = []
#         truths_ids[i] = []

#     for i in range(7):
#         maps = les[i].transform(les[i].classes_)
#         encode_dict = {}
#         for j in range(len(les[i].classes_)):
#             encode_dict[les[i].classes_[j]] = maps[j]
#         label_encode_dict.append(encode_dict)

#     for i, _label in enumerate(y_true):
#         for j in range(7):
#             if 'nan' in les[j].classes_:
#                 if _label[j] == label_encode_dict[j]['nan']: # Addressing 'nan' issues
#                     continue
#                 else:
#                     truths[j].append(_label[j])
#                     truths_ids[j].append(i)
#             else:
#                 truths[j].append(_label[j])
#                 truths_ids[j].append(i)

#     for i, _label in enumerate(y_pred):
#         for j in range(7):
#             if i in truths_ids[j]:
#                 preds[j].append(_label[j])
    
#     for i in range(7):
#         assert(len(truths[i])==len(preds[i]))

#     scores = {
#         'acc': [],
#         'f1': [],
#         'p_score': [],
#         'r_score': [],
#     }

#     print("Evaluating...")
#     all_classes = [0, 1]
#     for i in range(7):
#         print("class-id : {}".format(i+1))
#         acc, f1, p_score, r_score = scorer_indiv(truths[i], preds[i], all_classes)
#         for metric in scores:
#             scores[metric].append(eval(metric))        
#     scores_mean = copy.deepcopy(scores)
#     for i in scores_mean:
#         scores_mean[i]=np.mean(scores_mean[i])

#     return scores_mean, scores

# def evaluate(nmodel, test_dataloader, test_les, device):
#     y_preds, y_test = predict_labels(nmodel, test_dataloader, device)
#     scores_mean, scores = scorer(y_preds, y_test, test_les)
#     return scores_mean, scores

def inverse_transform(y):
    y = y.astype(str)
    # Convert encodings to hardcoded values
    for i in range(7):
        if i in [1, 2, 3, 4]:
            # Include nan
            col = np.copy(y[:,i])
            col[np.where(col == '0')] = 'no'
            col[np.where(col == '1')] = 'yes'
            col[np.where(col == '2')] = 'nan'
            y[:, i] = np.copy(col)
        else:
            col = np.copy(y[:,i])
            col[np.where(col == '0')] = 'no'
            col[np.where(col == '1')] = 'yes'
            y[:, i] = np.copy(col)

    return y


def generate_out_files_ml(y_preds, y_test):
    y_preds = inverse_transform(y_preds)
    y_test = inverse_transform(y_test)

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    np.savetxt("tmp/preds_tem.tsv", y_preds, delimiter="\t",fmt='%s')
    np.savetxt("tmp/gt_tem.tsv", y_test, delimiter="\t",fmt='%s')

    return y_preds, y_test

def generate_out_files(nmodel, test_dataloader, device):
    y_preds, y_test = predict_labels(nmodel, test_dataloader, device)
    
    y_preds = inverse_transform(y_preds)
    y_test = inverse_transform(y_test)

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    np.savetxt("tmp/preds_tem.tsv", y_preds, delimiter="\t",fmt='%s')
    np.savetxt("tmp/gt_tem.tsv", y_test, delimiter="\t",fmt='%s')

    return y_preds, y_test

def evaluate_model_ml(y_pred, y_test, return_files=False):
    y_preds, y_test = generate_out_files_ml(y_pred, y_test)

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

    if not return_files:
        return scores
    else:
        return scores, y_preds, y_test


def evaluate_model(nmodel, test_dataloader, device, return_files=False):
    y_preds, y_test = generate_out_files(nmodel, test_dataloader, device)

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

    if not return_files:
        return scores
    else:
        return scores, y_preds, y_test

def display_metrics(scores):
    print("--------------")
    print("---Mean Scores---")
    print('ACCURACY:', np.mean(scores['acc']))
    print('F1:', np.mean(scores['f1']))
    print('PRECISION:', np.mean(scores['p_score']))
    print('RECALL:', np.mean(scores['r_score']))

    print("\n---Classwise---")
    print('ACCURACY:', scores['acc'])
    print('F1:', scores['f1'])
    print('PRECISION:', scores['p_score'])
    print('RECALL:', scores['r_score'])

def get_model_embeddings(nmodel, dataloader, device):
    nmodel.eval()
    embeddings = []
    for i, batch in enumerate(dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():        
            emb = nmodel(sent_id, mask, get_embedding=True)
            emb = np.vstack([np.argmax(i.cpu().numpy(), axis=1) for i in emb]).T
            embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings

'''    
# scorer
def scorer(y_preds, y_test):
    # Averaged classwise accuracy
    accs = np.sum((y_preds==y_test).astype(int), axis=0)/len(y_test)*100
    total_accs = np.average(accs)
    return accs, total_accs

def evaluate(nmodel, test_dataloader, device):
    y_preds, y_test = predict_labels(nmodel, test_dataloader, device)
    classwise_acc, total_acc = scorer(y_preds, y_test)
    return classwise_acc, total_acc
'''

# Define training loop here #
def train(nmodel, training_dataloader, val_dataloader, device, epochs, lr=1e-5, loss_type="classwise_sum"):
    bert_params = nmodel.embeddings
    bert_named_params = ['embeddings.'+name_ for name_, param_ in bert_params.named_parameters()]
    model_named_params = [name_ for name_, param_ in nmodel.named_parameters()]
    other_named_params = [i for i in model_named_params if i not in bert_named_params]
    params = []

    for name, param in nmodel.named_parameters():
        if name in other_named_params:
            params.append(param)

    optimizer = AdamW(params,lr = lr,eps=1e-8)
    nmodel.train()

    loss_fn = None
    if loss_type == "classwise_sum":
        loss_fn = classwise_sum
    else:
        print("Loss {} Not Defined".format(loss_type))
        sys.exit(1)

    best_model = copy.deepcopy(nmodel)
    best_prec = 0
    for epoch_i in tqdm(range(0, epochs)):
        total_train_loss = 0
        nmodel.train()
        for step, batch in enumerate(training_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            nmodel.zero_grad()
            preds = nmodel(sent_id, mask)
            loss_val = loss_fn(preds, labels)
            total_train_loss += loss_val
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
        print('Total Train Loss =', total_train_loss)
        print('#############    Validation Set Stats')
        train_scores = evaluate_model(nmodel, training_dataloader, device)
        scores, val_preds, val_gt = evaluate_model(nmodel, val_dataloader, device, return_files=True)
        display_metrics(scores)

        # Log scores on wandb
        wandb.log({
            # Mean Scores
            'Training loss': total_train_loss,
            'Training Mean F1-Score': np.mean(train_scores['f1']),
            'Training Accuracy': np.mean(train_scores['acc']),
            'Training Mean Precision': np.mean(train_scores['p_score']),
            'Training Mean Recall': np.mean(train_scores['r_score']),

            'Validation Mean F1-Score': np.mean(scores['f1']),
            'Validation Accuracy': np.mean(scores['acc']),
            'Validation Mean Precision': np.mean(scores['p_score']),
            'Validation Mean Recall': np.mean(scores['r_score']),

            # Classwise scores
            # Train Set
            'Training Q1 F1 Score':train_scores['f1'][0],
            'Training Q2 F1 Score':train_scores['f1'][1],
            'Training Q3 F1 Score':train_scores['f1'][2],
            'Training Q4 F1 Score':train_scores['f1'][3],
            'Training Q5 F1 Score':train_scores['f1'][4],
            'Training Q6 F1 Score':train_scores['f1'][5],
            'Training Q7 F1 Score':train_scores['f1'][6],

            # Validation Set
            'Validation Q1 F1 Score':scores['f1'][0],
            'Validation Q2 F1 Score':scores['f1'][1],
            'Validation Q3 F1 Score':scores['f1'][2],
            'Validation Q4 F1 Score':scores['f1'][3],
            'Validation Q5 F1 Score':scores['f1'][4],
            'Validation Q6 F1 Score':scores['f1'][5],
            'Validation Q7 F1 Score':scores['f1'][6]
        }, step=epoch_i)

        if np.mean(scores['p_score']) > best_prec:
            best_model = copy.deepcopy(nmodel)
            best_prec = np.mean(scores['p_score'])

            # Save model to wandb
            # wandb.save('checkpoints_best_val.pt')
            # nmodel.save(os.path.join(wandb.run.dir, "best_val.pt"))
            # torch.save(nmodel.state_dict(), os.path.join(wandb.run.dir, "best_val.pth"))

            np.savetxt(os.path.join(wandb.run.dir, "best_val_preds.tsv"), val_preds, delimiter="\t",fmt='%s')
            np.savetxt(os.path.join(wandb.run.dir, "best_val_grount_truth.tsv"), val_gt, delimiter="\t",fmt='%s')

    return best_model

def train_v2(nmodel, training_dataloader, val_dataloader, device, epochs = 4, lr1=2e-5, lr2=1e-4, loss_type="classwise_sum"):
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
    else:
        print("Loss {} Not Defined".format(loss_type))
        sys.exit(1)

    # Load class weights for loss function
    wts = []
    for i in range(7):
        wts.append(torch.Tensor(np.load('data/class_weights/q' + str(i+1) + '.npy')).to(device))

    best_model = copy.deepcopy(nmodel)
    best_prec = 0
    for epoch_i in tqdm(range(0, epochs)):
        total_train_loss = 0
        nmodel.train()
        for step, batch in enumerate(training_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch

            # print(sent_id.shape)
            # print("\n\n\n")
            # print(mask.shape)

            ypreds = nmodel(sent_id, mask)
            loss = loss_fn(ypreds, labels, wts)
            # if step%50==0:
            #     loss_val = total_train_loss/(step+1.00)
            #     print('Loss = '+loss_val)
            #     # wandb.log({"loss":loss_val})

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
            'Training loss': total_train_loss,
            'Training Mean F1-Score': np.mean(train_scores['f1']),
            'Training Accuracy': np.mean(train_scores['acc']),
            'Training Mean Precision': np.mean(train_scores['p_score']),
            'Training Mean Recall': np.mean(train_scores['r_score']),

            'Validation Mean F1-Score': np.mean(scores['f1']),
            'Validation Accuracy': np.mean(scores['acc']),
            'Validation Mean Precision': np.mean(scores['p_score']),
            'Validation Mean Recall': np.mean(scores['r_score']),

            # Classwise scores
            # Train Set
            'Training Q1 F1 Score':train_scores['f1'][0],
            'Training Q2 F1 Score':train_scores['f1'][1],
            'Training Q3 F1 Score':train_scores['f1'][2],
            'Training Q4 F1 Score':train_scores['f1'][3],
            'Training Q5 F1 Score':train_scores['f1'][4],
            'Training Q6 F1 Score':train_scores['f1'][5],
            'Training Q7 F1 Score':train_scores['f1'][6],

            # Validation Set
            'Validation Q1 F1 Score':scores['f1'][0],
            'Validation Q2 F1 Score':scores['f1'][1],
            'Validation Q3 F1 Score':scores['f1'][2],
            'Validation Q4 F1 Score':scores['f1'][3],
            'Validation Q5 F1 Score':scores['f1'][4],
            'Validation Q6 F1 Score':scores['f1'][5],
            'Validation Q7 F1 Score':scores['f1'][6]
        }, step=epoch_i)

        if np.mean(scores['p_score']) > best_prec:
            best_model = copy.deepcopy(nmodel)
            best_prec = np.mean(scores['p_score'])

            # Save model to wandb
            # wandb.save('checkpoints_best_val.pt')
            # nmodel.save(os.path.join(wandb.run.dir, "best_val.pt"))
            # torch.save(nmodel.state_dict(), os.path.join(wandb.run.dir, "best_val.pth"))

            np.savetxt(os.path.join(wandb.run.dir, "best_val_preds.tsv"), val_preds, delimiter="\t",fmt='%s')
            np.savetxt(os.path.join(wandb.run.dir, "best_val_grount_truth.tsv"), val_gt, delimiter="\t",fmt='%s')


    
    return best_model