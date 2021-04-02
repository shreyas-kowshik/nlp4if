import torch.nn as nn
import transformers
from transformers import AutoModel,BertModel,  BertTokenizerFast
import torch

class BERTBasic(nn.Module):
    def __init__(self, freeze_bert_params=True):
      super(BERTBasic, self).__init__()
      self.embeddings = AutoModel.from_pretrained('bert-base-uncased')#, output_hidden_states = True)

      if freeze_bert_params:
      	for param in self.embeddings.parameters():
      		param.requires_grad=False

      self.dropout = nn.Dropout(0.1)
      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(768,512)
      self.fc2 = nn.Linear(512,512)

      # softmax activation functions
      self.out_shared = nn.Linear(512, 256)
      self.out1 = nn.Linear(256, 2)
      self.out2 = nn.Linear(256, 3)
      self.out3 = nn.Linear(256, 3)
      self.out4 = nn.Linear(256, 3)
      self.out5 = nn.Linear(256, 3)
      self.out6 = nn.Linear(512, 2)
      self.out7 = nn.Linear(512, 2)
      

    #define the forward pass
    def forward(self, sent_id, mask):
      # Bert
      cls_hs = self.embeddings(sent_id, attention_mask=mask)[1]

      # Initial layers
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      
      # Share out1 to out5 with initial weights since output depends on output of out1
      shared_x = self.out_shared(x)
      out1 = self.out1(shared_x)
      out2 = self.out2(shared_x)
      out3 = self.out3(shared_x)
      out4 = self.out4(shared_x)
      out5 = self.out5(shared_x)
      out6 = self.out6(x)
      out7 = self.out7(x)

      return [out1, out2, out3, out4, out5, out6, out7]


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    # Input shape : (32(B), 56(SEQ_LEN), 512(INPUT DIM))
    # Output shape : (32, 512)
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = (a - torch.min(a, 1, keepdim=True)[0])
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class BERTAttention(nn.Module):
    def __init__(self, freeze_bert_params=True):
      super(BERTAttention, self).__init__()
      self.embeddings = AutoModel.from_pretrained('bert-base-uncased')#, output_hidden_states = True)

      if freeze_bert_params:
        for param in self.embeddings.parameters():
          param.requires_grad=False

      self.dropout = nn.Dropout(0.1)
      self.lstm = nn.LSTM(768, 256, bidirectional=True, batch_first=True)
      self.attention_layer = Attention(512, 56) # max_len in hard coded here

      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(512,512)
      self.fc2 = nn.Linear(512,512)

      # softmax activation functions
      self.out_shared = nn.Linear(512, 256)
      self.out1 = nn.Linear(256, 2)
      self.out2 = nn.Linear(256, 3)
      self.out3 = nn.Linear(256, 3)
      self.out4 = nn.Linear(256, 3)
      self.out5 = nn.Linear(256, 3)
      self.out6 = nn.Linear(512, 2)
      self.out7 = nn.Linear(512, 2)


    #define the forward pass
    def forward(self, sent_id, mask):
      # Bert
      sequence_output = self.embeddings(sent_id, attention_mask=mask)[0]
      print("Sequence Output Shape : {}".format(sequence_output.shape))
      # LSTM, Attention
      lstm_layer, _ = self.lstm(sequence_output)
      print("LSTM Out Shape : {}".format(lstm_layer.shape))
      attn_layer = self.attention_layer(lstm_layer)
      print("Attention Shape : {}".format(attn_layer.shape))

      # Initial layers
      x = self.fc1(attn_layer)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)

      # Share out1 to out5 with initial weights since output depends on output of out1
      shared_x = self.out_shared(x)
      out1 = self.out1(shared_x)
      out2 = self.out2(shared_x)
      out3 = self.out3(shared_x)
      out4 = self.out4(shared_x)
      out5 = self.out5(shared_x)
      out6 = self.out6(x)
      out7 = self.out7(x)

      return [out1, out2, out3, out4, out5, out6, out7]
