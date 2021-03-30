import numpy as np
import copy
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.losses import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support
import wandb

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


def scorer_indiv(truths, submitted, all_classes):
    acc = accuracy_score(truths, submitted)
    f1 = f1_score(truths, submitted, labels=all_classes, average='weighted')
    p_score = precision_score(truths, submitted, labels=all_classes, average='weighted')
    r_score = recall_score(truths, submitted, labels=all_classes, average='weighted')
    return acc, f1, p_score, r_score

def scorer(y_true, y_pred):
    truths = {}
    truths_ids = {}
    preds = {}

    for i in range(7):
        truths[i] = []
        preds[i] = []
        truths_ids[i] = []

    for i, _label in enumerate(y_true):
        for j in range(7):
            if _label[j]!=2 :
                truths[j].append(_label[j])
                truths_ids[j].append(i)

    for i, _label in enumerate(y_pred):
        for j in range(7):
            if i in truths_ids[j]:
                preds[j].append(_label[j])
    
    for i in range(7):
        assert(len(truths[i])==len(preds[i]))

    scores = {
        'acc': [],
        'f1': [],
        'p_score': [],
        'r_score': [],
    }

    print("Evaluating...")
    all_classes = [0, 1]
    for i in range(7):
        print("class-id : {}".format(i+1))
        acc, f1, p_score, r_score = scorer_indiv(truths[i], preds[i], all_classes)
        for metric in scores:
            scores[metric].append(eval(metric))        
    scores_mean = copy.deepcopy(scores)
    for i in scores_mean:
        scores_mean[i]=np.mean(scores_mean[i])

    return scores_mean, scores

def evaluate(nmodel, test_dataloader, device):
    y_preds, y_test = predict_labels(nmodel, test_dataloader, device)
    scores_mean, scores = scorer(y_preds, y_test)
    return scores_mean, scores

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
def train(model, dataloader, val_dataloader, device, num_epochs, lr=1e-5, loss_type="classwise_sum"):
    optimizer = AdamW(model.parameters(),lr = lr)
    model.train()

    loss_fn = None
    if loss_type == "classwise_sum":
        loss_fn = classwise_sum
    else:
        print("Loss {} Not Defined".format(loss_type))
        sys.exit(1)

    for epoch in range(num_epochs):
        for i,batch in enumerate(dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            model.zero_grad()
            preds = model(sent_id, mask)
            loss_val = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
        print("Epoch : {}  Loss : {}".format(epoch, loss_val.cpu()))
        if(epoch%10==9):
            scores, scores_mean = evaluate(model, val_dataloader, device)
            print('Mean scores: ', scores_mean, '\nClass wise scores: ', scores)
    return model

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
    
    optimizer1 = AdamW(bert_params.parameters(), lr=2e-5, eps = 1e-8)
    optimizer2 = AdamW(params, lr=1e-4, eps = 1e-8)
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

    for epoch_i in tqdm(range(0, epochs)):
        total_train_loss = 0
        nmodel.train()
        for step, batch in enumerate(training_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            ypreds = nmodel(sent_id, mask)
            loss = loss_fn(ypreds, labels)
            if step%50==0:
                loss_val = total_train_loss/(step+1.00)
                print('Loss = '+str(total_train_loss/(step+1.00)))
                # wandb.log({"loss":loss_val})

            total_train_loss += loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_params.parameters(), 1.0)
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

        print('Total Train Loss = {total_train_loss}')
        print('#############    Validation Set Stats')
        scores_mean, scores = evaluate(nmodel, val_dataloader, device)
        print('Mean scores: ', scores_mean, '\nClass wise scores: ', scores)

    return nmodel