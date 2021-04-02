import torch.nn as nn
import transformers
from transformers import AutoModel,BertModel,  BertTokenizerFast
import torch
import math
import torch.nn.functional as F

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    # This is a seq_lenxseq_len matrix, shape : (bs, heads, q_seq_len, k_seq_len)
    # print("Scores : {}".format(scores.shape))
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output

class LinearBlock(nn.Module):
  def __init__(self, in_dim, out_dim, use_bn=False):
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

  def forward(self, x):
    if self.use_bn:
      return self.relu(self.bn(self.linear(x)))
    else:
      return self.relu(self.linear(x))




class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1, debug=False):
        super(MultiHeadAttention, self).__init__()
        
        self.debug=debug
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    # q : queries, k : keys, v:values, here all values will be our 8 tensors
    # size of each : (Batch_Size, Seq_len, Input Dim)

    # (32,8,3*512)
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        if self.debug:
          print("Q, K, V : {}, {}, {}".format(q.shape, k.shape, v.shape))
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        if self.debug:
          print("After linear layer")
          print("Q, K, V : {}, {}, {}".format(q.shape, k.shape, v.shape))
        
        # transpose to get dimensions bs * heads * seq_len * d_k
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next

        if self.debug:
          print("After transpose Q, K, V : {}, {}, {}".format(q.shape, k.shape, v.shape))

        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        if self.debug:
          print("Scores : {}".format(scores.shape))

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        if self.debug:
          print("Concat : {}".format(concat.shape))
        
        output = self.out(concat)

        if self.debug:
          print("Output : {}".format(output.shape))
    
        return output



class BERTAttentionClasswise(nn.Module):
    # TODO : Add dropout prob to arparse
    def __init__(self, freeze_bert_params=True, dropout_prob=0.1, num_heads=3):
      super(BERTAttentionClasswise, self).__init__()
      self.embeddings = AutoModel.from_pretrained('bert-base-uncased')#, output_hidden_states = True)

      if freeze_bert_params:
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

      self.fc1 = LinearBlock(768,512)
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
    def forward(self, sent_id, mask):
      # Bert
      bert_output = self.embeddings(sent_id, attention_mask=mask)[1]

      x = self.fc1(bert_output)
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

      # print(attn_inp.shape)

      # Apply attention
      # attn_out = self.attn(attn_inp, attn_inp, attn_inp)

      # print("Attn Out : {}".format(attn_out.shape))

      out1 = self.out1(attn_out1)
      out2 = self.out2(attn_out2)
      out3 = self.out3(attn_out3)
      out4 = self.out4(attn_out4)
      out5 = self.out5(attn_out5)
      out6 = self.out6(attn_out6)
      out7 = self.out7(attn_out7)

      # print(out1.shape)
      # print(out2.shape)
      # print(out3.shape)
      # print(out4.shape)
      # print(out5.shape)
      # print(out6.shape)
      # print(out7.shape)

      return [out1, out2, out3, out4, out5, out6, out7]
