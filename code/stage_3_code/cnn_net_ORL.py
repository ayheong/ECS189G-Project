import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)       # 112x92 → 108x88
        self.pool = nn.MaxPool2d(2, 2)         # 108x88 → 54x44
        self.fc1 = nn.Linear(32 * 54 * 44, 512)
        self.fc2 = nn.Linear(512, 40)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → 32x54x44
        x = torch.flatten(x, 1)                # → 32*54*44
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
