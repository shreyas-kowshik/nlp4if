import torch.nn as nn
import transformers
from transformers import AutoModel,BertModel,  BertTokenizerFast
import torch
import math
import torch.nn.functional as F
from BertAttentionClasswise import *



class BERTAttentionIncorp(nn.Module):
    def __init__(self, incorp_dim, freeze_bert_params=True, dropout_prob=0.1, num_heads=3, bert_base='bert-base-uncased'):
      print("BERTAttentionIncorp Being Used!\n\n\n")

      super(BERTAttentionIncorp, self).__init__()
      self.embeddings = AutoModel.from_pretrained(bert_base)#, output_hidden_states = True)

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

      embedding_dim=768
      if bert_base=='bert-large-cased':
        embedding_dim=1024

      self.fc1 = LinearBlock(1024,512)
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

      # Incorporation Branch #
      self.incorp_dim = incorp_dim
      self.inc_fc1 = LinearBlock(self.incorp_dim, 256)
      self.inc_fc2 = LinearBlock(256, 512)
      self.inc_dropout = nn.Dropout(dropout_prob)
      self.inc_fc3 = LinearBlock(512, 512)

      # Penultimate layers
      self.out1 = nn.Linear(self.num_heads * 256 + 512, 2)
      self.out2 = nn.Linear(self.num_heads * 256 + 512, 3)
      self.out3 = nn.Linear(self.num_heads * 256 + 512, 3)
      self.out4 = nn.Linear(self.num_heads * 256 + 512, 3)
      self.out5 = nn.Linear(self.num_heads * 256 + 512, 3)
      self.out6 = nn.Linear(self.num_heads * 256 + 512, 2)
      self.out7 = nn.Linear(self.num_heads * 256 + 512, 2)


    #define the forward pass
    def forward(self, sent_id, mask, inc_feats, get_embedding=False):
      # Bert
      bert_output = self.embeddings(sent_id, attention_mask=mask)[1]

      # print(bert_output.shape)
      # print("\n\n\n")

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

      if get_embedding:
        out_cat = torch.cat([attn_out1, attn_out2, attn_out3, attn_out4, attn_out5, attn_out6, attn_out7], 1)
        return out_cat

      # Late Incorporation Branch #
      inc_out1 = self.inc_fc1(inc_feats)
      inc_out2 = self.inc_fc2(inc_out1)
      inc_out2 = self.inc_dropout(inc_out2)
      inc_out3 = self.inc_fc3(inc_out2)

      attn_out1 = torch.cat([attn_out1, inc_out3], 1)
      attn_out2 = torch.cat([attn_out2, inc_out3], 1)
      attn_out3 = torch.cat([attn_out3, inc_out3], 1)
      attn_out4 = torch.cat([attn_out4, inc_out3], 1)
      attn_out5 = torch.cat([attn_out5, inc_out3], 1)
      attn_out6 = torch.cat([attn_out6, inc_out3], 1)
      attn_out7 = torch.cat([attn_out7, inc_out3], 1)

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
