import pandas as pd
import os
import torch.nn as nn
import torch
import torch.optim as optim
from utils.preprocess import *

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
    def forward(self, sent_id, mask):
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
      out6 = self.out6(x)
      out7 = self.out7(x)

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

