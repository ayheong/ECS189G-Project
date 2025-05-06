import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)       # 112x92 → 108x88
        self.conv2 = nn.Conv2d(32, 64, 5)      # 108x88 → 104x84
        self.pool = nn.MaxPool2d(2, 2)         # Apply pooling after each conv
        self.fc1 = nn.Linear(64 * 25 * 20, 512)
        self.fc2 = nn.Linear(512, 40)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → 54x44
        x = self.pool(F.relu(self.conv2(x)))   # → 26x21
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
