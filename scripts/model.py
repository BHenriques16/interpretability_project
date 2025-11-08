import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_class=40):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):                
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 32 * 32)          
        x = F.relu(self.fc1(x))                
        x = torch.sigmoid(self.fc2(x))        
        return x                    