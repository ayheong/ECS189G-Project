import pickle
import torch
import numpy as np

with open('data/stage_3_data/ORL', 'rb') as f:
    data = pickle.load(f)

all_images = []

for split in ['train', 'test']:
    for instance in data[split]:
        img = instance['image'][:, :, 0]  # use just one channel (grayscale)
        all_images.append(img.astype(np.float32) / 255.0)  # normalize to [0, 1]

all_images = np.stack(all_images)  # shape: (N, H, W)

mean = np.mean(all_images)
std = np.std(all_images)

print(f'ORL grayscale mean: {mean:.4f}, std: {std:.4f}')
