import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_class=40):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):                      
        x = self.pool(F.relu(self.conv1(x)))  # passa pela conv1 + relu + pooling
        x = self.pool(F.relu(self.conv2(x)))  # passa pela conv2 + relu + pooling
        x = x.view(-1, 64 * 32 * 32)          # achata para vetor (flatten)
        x = F.relu(self.fc1(x))               # primeira camada totalmente conectada + relu
        x = torch.sigmoid(self.fc2(x))        # camada final + sigmoid para probabilidades
        return x                              # output final