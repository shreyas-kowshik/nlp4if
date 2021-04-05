import pandas as pd
import os
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.preprocess import *
from utils.train_utils import *
from scorer.main import *

def evaluate_glove(iterator, model, device, return_files=False):
    y_pred, y_test=predict_glove(iterator, model, device)
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

def predict_glove(iterator, model, device):
    model.eval()
    all_y = []
    all_y_hat = []
    for ni, batch in enumerate(iterator):
        y = torch.stack([batch.q1_label,
                            batch.q2_label,
                            batch.q3_label,
                            batch.q4_label,
                            batch.q5_label,
                            batch.q6_label,
                            batch.q7_label,
                            ],dim=1).to(device)
        y_hat = model(batch.tweet_text.to(device))
        all_y.append(y)
        all_y_hat.append(np.vstack([np.argmax(i.detach().cpu().numpy(), axis=1) for i in y_hat]).T)
        # q=np.vstack([np.argmax(i.detach().cpu().numpy(), axis=1) for i in y_hat]).T

    y_preds = np.vstack(all_y_hat)
    y_test = np.vstack(all_y)
    return y_preds, y_test

def train_n_epochs(model, train_iterator, val_iterator, n, lr, wd, cw, device):
    criterion = classwise_sum
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(n):
        model, loss = fit_epoch(train_iterator, model, optimizer, criterion, cw, device)
        train_scores=evaluate_glove(train_iterator, model, device)
        val_scores=evaluate_glove(val_iterator, model, device)
        print('Epoch:', epoch, 'Mean F1 train', np.mean(train_scores['f1']), 'Val F1 train', np.mean(val_scores['f1']))
    return model


def fit_epoch(iterator, model, optimizer, criterion, cw, device):
    train_loss = 0
    train_acc = 0
    model.train()
    all_y = []
    all_y_hat = []
    for ni, batch in enumerate(iterator):
        optimizer.zero_grad()
        y = torch.stack([batch.q1_label,
                         batch.q2_label,
                         batch.q3_label,
                         batch.q4_label,
                         batch.q5_label,
                         batch.q6_label,
                         batch.q7_label,
                         ],dim=1).to(device)
        y_hat = model(batch.tweet_text.to(device))
        loss = criterion(y_hat, y, cw)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        all_y.append(y)
        all_y_hat.append(y_hat)
    return model, loss

def convert_dataframe(df_path, file_name):
    if os.path.isfile('data_glove/'+file_name):
        print(file_name, 'already exists!')
        return
    df=pd.read_csv(df_path, sep='\t')
    df=df.dropna(subset=['q7_label', 'q6_label'])  
    sentences, labels, train_les = process_data(df_path)
    labels=np.array(labels)
    for i in range(7):
        df['q'+str(i+1)+'_label']=labels[:,i]
    df.to_csv('data_glove/'+file_name, sep='\t', index=False)

class GloveNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, embeddings, text_length, lstm_hidden_size):
      super(GloveNet, self).__init__()

      self.embeddings = nn.Embedding(vocab_size,embedding_dim,pad_idx)
      self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
      self.max_pool = nn.MaxPool2d((text_length,1))
      self.fc1 = nn.Linear(lstm_hidden_size, 50)

      # softmax activation functions
      self.out1 = nn.Linear(50, 2)
      self.out2 = nn.Linear(50, 3)
      self.out3 = nn.Linear(50, 3)
      self.out4 = nn.Linear(50, 3)
      self.out5 = nn.Linear(50, 3)
      self.out6 = nn.Linear(50, 2)
      self.out7 = nn.Linear(50, 2)
      
    #define the forward pass
    def forward(self, text):
      a1 = self.embeddings(text)
      a2 = self.lstm(a1)[0]
      a3 = self.max_pool(a2).squeeze(1)
      shared_x = F.relu(self.fc1(a3))
      
      # Share out1 to out5 with initial weights since output depends on output of out1
      out1 = self.out1(shared_x)
      out2 = self.out2(shared_x)
      out3 = self.out3(shared_x)
      out4 = self.out4(shared_x)
      out5 = self.out5(shared_x)
      out6 = self.out6(shared_x)
      out7 = self.out7(shared_x)

      return [out1, out2, out3, out4, out5, out6, out7]

class NNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, embeddings, text_length, lstm_hidden_size):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size,embedding_dim,pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.max_pool = nn.MaxPool2d((text_length,1))
        self.fc1 = nn.Linear(lstm_hidden_size, 50)
        self.fc2 = nn.Linear(50, output_dim)

    def forward(self, text):
        a1 = self.embeddings(text)
        a2 = self.lstm(a1)[0]
        a3 = self.max_pool(a2).squeeze(1)
        a4 = F.relu(self.fc1(a3))
        a5 = self.fc2(a4)
        return a5
