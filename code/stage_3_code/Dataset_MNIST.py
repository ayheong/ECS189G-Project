import torch
from torch.utils.data import Dataset

class Dataset_MNIST(Dataset):
    def __init__(self, data_list, normalize=True, mean=0.1307, std=0.3081):
        self.data_list = data_list
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(1, 1, 1)
        self.std = torch.tensor(std).view(1, 1, 1)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image = self.data_list[idx]['image']
        label = self.data_list[idx]['label']

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, H, W)

        if self.normalize:
            image = (image - self.mean) / self.std

        return image, label
