import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)       # 28x28 → 24x24
        self.pool = nn.MaxPool2d(2, 2)         # 24x24 → 12x12
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → 32x12x12
        x = torch.flatten(x, 1)                # → 4608
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
