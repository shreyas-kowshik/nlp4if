import torch
import torch.nn as nn
from config.config import *

def classwise_sum(outputs, targets, weights, device=torch.device(device_name)):
    num_labels = targets.shape[1]

    criterion = [nn.CrossEntropyLoss(weight=weights[i]) for i in range(7)]
    loss = torch.tensor(0.0).to(device)
    for i in range(targets.shape[1]):
        # Add extra weight to 1
        if i == 0:
            loss += (3.0 * criterion[i](outputs[i], targets[:, i]))
        else:
            loss += criterion[i](outputs[i], targets[:, i])
    
    return loss
