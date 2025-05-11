import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)        # 28x28 → 24x24
        self.pool = nn.MaxPool2d(2, 2)          # 24x24 → 12x12
        self.conv2 = nn.Conv2d(32, 64, 3)       # 12x12 → 10x10
        self.fc1 = nn.Linear(64 * 5 * 5, 128)   # After 2nd pool: 10x10 → 5x5
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # → 32x12x12
        x = self.pool(F.relu(self.conv2(x)))    # → 64x5x5
        x = torch.flatten(x, 1)                 # → 1600
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
