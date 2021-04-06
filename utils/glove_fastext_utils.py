import pandas as pd
import os
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.preprocess import *
from utils.train_utils import *
from scorer.main import *
from BertAttentionClasswise import *

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
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, embeddings_weight, text_length, lstm_hidden_size):
      super(GloveNet, self).__init__()

      self.embeddings = nn.Embedding(vocab_size,embedding_dim,pad_idx)
      embeddings.weight.data.copy_(torch.from_numpy(embeddings_weight))
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

class GloveFasttextAttentionClasswise(nn.Module):
    # TODO : Add dropout prob to arparse
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, embeddings_weight, freeze_emb_params=True, dropout_prob=0.1, num_heads=3):
      super(GloveFasttextAttentionClasswise, self).__init__()
      self.embeddings = nn.Embedding(vocab_size,embedding_dim,pad_idx)
      embeddings.weight.data.copy_(torch.from_numpy(embeddings_weight))
      
      if freeze_emb_params:
        for param in self.embeddings.parameters():
          param.requires_grad=False

      self.num_heads = num_heads

      self.dropout_common = nn.Dropout(dropout_prob)
      self.dropout1 = nn.Dropout(dropout_prob)
      self.dropout2 = nn.Dropout(dropout_prob)
      self.dropout3 = nn.Dropout(dropout_prob)
      self.dropout4 = nn.Dropout(dropout_prob)
      self.dropout5 = nn.Dropout(dropout_prob)
      self.dropout6 = nn.Dropout(dropout_prob)
      self.dropout7 = nn.Dropout(dropout_prob)

      self.fc1 = LinearBlock(embedding_dim,512)
      self.fc2 = LinearBlock(512,512)

      self.fc_out1 = LinearBlock(512,512)
      self.fc_out2 = LinearBlock(512,512)
      self.fc_out3 = LinearBlock(512,512)
      self.fc_out4 = LinearBlock(512,512)
      self.fc_out5 = LinearBlock(512,512)
      self.fc_out6 = LinearBlock(512,512)
      self.fc_out7 = LinearBlock(512,512)

      self.fc_out1_2 = LinearBlock(512,256)
      self.fc_out2_2 = LinearBlock(512,256)
      self.fc_out3_2 = LinearBlock(512,256)
      self.fc_out4_2 = LinearBlock(512,256)
      self.fc_out5_2 = LinearBlock(512,256)
      self.fc_out6_2 = LinearBlock(512,256)
      self.fc_out7_2 = LinearBlock(512,256)

      self.attn1 = MultiHeadAttention(self.num_heads, self.num_heads * 256)
      self.attn2 = MultiHeadAttention(self.num_heads, self.num_heads * 256)
      self.attn3 = MultiHeadAttention(self.num_heads, self.num_heads * 256)
      self.attn4 = MultiHeadAttention(self.num_heads, self.num_heads * 256)
      self.attn5 = MultiHeadAttention(self.num_heads, self.num_heads * 256)
      self.attn6 = MultiHeadAttention(self.num_heads, self.num_heads * 256)
      self.attn7 = MultiHeadAttention(self.num_heads, self.num_heads * 256)

      # Penultimate layers
      self.out1 = nn.Linear(self.num_heads * 256, 2)
      self.out2 = nn.Linear(self.num_heads * 256, 3)
      self.out3 = nn.Linear(self.num_heads * 256, 3)
      self.out4 = nn.Linear(self.num_heads * 256, 3)
      self.out5 = nn.Linear(self.num_heads * 256, 3)
      self.out6 = nn.Linear(self.num_heads * 256, 2)
      self.out7 = nn.Linear(self.num_heads * 256, 2)


    #define the forward pass
    def forward(self, text):
      emb_output = self.embeddings(text)

      x = self.fc1(emb_output)
      x = self.dropout_common(x)
      x = self.fc2(x)

      # Individual Processing
      x1 = self.fc_out1(x)
      x1 = self.dropout1(x1)
      x1 = self.fc_out1_2(x1)

      x2 = self.fc_out2(x)
      x2 = self.dropout2(x2)
      x2 = self.fc_out2_2(x2)

      x3 = self.fc_out3(x)
      x3 = self.dropout3(x3)
      x3 = self.fc_out3_2(x3)

      x4 = self.fc_out4(x)
      x4 = self.dropout4(x4)
      x4 = self.fc_out4_2(x4)

      x5 = self.fc_out5(x)
      x5 = self.dropout5(x5)
      x5 = self.fc_out5_2(x5)

      x6 = self.fc_out6(x)
      x6 = self.dropout6(x6)
      x6 = self.fc_out6_2(x6)

      x7 = self.fc_out7(x)
      x7 = self.dropout7(x7)
      x7 = self.fc_out7_2(x7)

      # Attention
      # Prepare input
      x1 = torch.cat(self.num_heads*[x1], 1).unsqueeze(1)
      x2 = torch.cat(self.num_heads*[x2], 1).unsqueeze(1)
      x3 = torch.cat(self.num_heads*[x3], 1).unsqueeze(1)
      x4 = torch.cat(self.num_heads*[x4], 1).unsqueeze(1)
      x5 = torch.cat(self.num_heads*[x5], 1).unsqueeze(1)
      x6 = torch.cat(self.num_heads*[x6], 1).unsqueeze(1)
      x7 = torch.cat(self.num_heads*[x7], 1).unsqueeze(1)

      attn_cat_inp = torch.cat([x1, x2, x3, x4, x5, x6, x7], 1)
      attn_out1 = self.attn1(x1, attn_cat_inp, attn_cat_inp).reshape(-1, self.num_heads * 256)
      attn_out2 = self.attn2(x2, attn_cat_inp, attn_cat_inp).reshape(-1, self.num_heads * 256)
      attn_out3 = self.attn3(x3, attn_cat_inp, attn_cat_inp).reshape(-1, self.num_heads * 256)
      attn_out4 = self.attn4(x4, attn_cat_inp, attn_cat_inp).reshape(-1, self.num_heads * 256)
      attn_out5 = self.attn5(x5, attn_cat_inp, attn_cat_inp).reshape(-1, self.num_heads * 256)
      attn_out6 = self.attn6(x6, attn_cat_inp, attn_cat_inp).reshape(-1, self.num_heads * 256)
      attn_out7 = self.attn7(x7, attn_cat_inp, attn_cat_inp).reshape(-1, self.num_heads * 256)

      out1 = self.out1(attn_out1)
      out2 = self.out2(attn_out2)
      out3 = self.out3(attn_out3)
      out4 = self.out4(attn_out4)
      out5 = self.out5(attn_out5)
      out6 = self.out6(attn_out6)
      out7 = self.out7(attn_out7)

      return [out1, out2, out3, out4, out5, out6, out7]
