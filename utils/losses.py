import torch
import torch.nn as nn
from config.config import *

def classwise_sum(outputs, targets, device=torch.device(device_name)):
    num_labels = targets.shape[1]
    criterion = [nn.CrossEntropyLoss() for i in range(7)]
    loss = torch.tensor(0.0).to(device)
    for i in range(targets.shape[1]):
        loss += criterion[i](outputs[i], targets[:, i])
    
    return loss
