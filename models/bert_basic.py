import torch.nn as nn
import transformers
from transformers import AutoModel,BertModel,  BertTokenizerFast

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

      # [DONE] TODO : Values here should be for 2 classes below according to PS README, but train-data has 3 values in it
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
