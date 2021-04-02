import torch; import torch.nn as nn;
from models.BertAttentionClasswise import *

bs=32
seq=8
heads=3
dim=512
q = torch.ones(bs, heads * dim)
k = torch.ones(bs, seq, heads * dim)
model = MultiHeadAttention(heads, heads * dim)
out = model(q, k, k)