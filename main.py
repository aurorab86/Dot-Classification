import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from data_load import LoadData
from data_divide import Divide

import toy_data

data = LoadData(num=3) # num: class number
xtrain, ytrain, xtest, ytest = Divide(data)

def accuracy(pred, turth):
    predicted_labels = torch.argmax(pred, axis = 1)
    correct = (predicted_labels == turth).float()
    accuracy = correct.mean().item()

    return accuracy


class Model(nn.Module):
    def __init__(self, num_neurons, num_classes):
        super(Model, self).__init__()

        self.dense1 = nn.Linear(2, num_neurons)
        self.dense2 = nn.Linear(num_neurons, num_classes)

        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)

        return x
    
    def parameters(self):
        return tuple(self.dense1.parameters()) + tuple(self.dense2.parameters())
    
    def load_parameter(self, w1, b1, w2):
        self.dense1.weight.data = w1
        self.dense1.bias.data = b1
        self.dense2.weight.data = w2


