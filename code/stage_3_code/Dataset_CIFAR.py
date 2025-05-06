import torch
from torch.utils.data import Dataset

class Dataset_CIFAR(Dataset):
    def __init__(self, data_list, normalize=True, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
        self.data_list = data_list
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image = self.data_list[idx]['image']
        label = self.data_list[idx]['label']

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)

        if self.normalize:
            image = (image - self.mean) / self.std

        return image, label
