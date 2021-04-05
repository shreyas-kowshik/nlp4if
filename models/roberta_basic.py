import torch.nn as nn
import transformers
from transformers import AutoModel, RobertaTokenizer, RobertaModel
import torch

class LinearBlock(nn.Module):
  def __init__(self, in_dim, out_dim, use_bn=True):
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

  def forward(self, x):
    if self.use_bn:
      return self.relu(self.bn(self.linear(x)))
    else:
      return self.relu(self.linear(x))

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

class ROBERTaAttention(nn.Module):
    def __init__(self, freeze_bert_params=True, dropout_prob=0.1, bert_base='roberta-base'):
      super(ROBERTaAttention, self).__init__()
      self.embeddings = RobertaModel.from_pretrained(bert_base)

      if freeze_bert_params:
        for param in self.embeddings.parameters():
          param.requires_grad=False

      self.dropout_common = nn.Dropout(dropout_prob)

      embedding_dim=768
      if bert_base=='bert-large-cased':
        embedding_dim=1024

      self.lstm = nn.LSTM(embedding_dim, 256, bidirectional=True, batch_first=True)
      self.attention_layer = Attention(512, 56) # max_len in hard coded here

      self.fc1 = LinearBlock(512,512)
      self.fc2 = LinearBlock(512,512)

      # softmax activation functions
      self.out_shared = LinearBlock(512, 256)
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
      #print("Sequence Output Shape : {}".format(sequence_output.shape))
      # LSTM, Attention
      lstm_layer, _ = self.lstm(sequence_output)
      #print("LSTM Out Shape : {}".format(lstm_layer.shape))
      attn_layer = self.attention_layer(lstm_layer)
      #print("Attention Shape : {}".format(attn_layer.shape))

      # Initial layers
      x = self.fc1(attn_layer)
      x = self.dropout_common(x)
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
