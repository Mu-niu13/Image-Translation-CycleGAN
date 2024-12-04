# lib/data_loader.py

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

class MonetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, img) for img in os.listdir(root_dir)
            if img.endswith('.jpg') or img.endswith('.png')
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_dataloader(dataset_name, root_dir, transform, batch_size, shuffle=True):
    if dataset_name.lower() == 'celeba':
        dataset = datasets.CelebA(
            root=root_dir, split='train', download=True, transform=transform
        )
    elif dataset_name.lower() == 'lsun':
        dataset = datasets.LSUN(
            root=root_dir, classes='train', transform=transform
        )
    elif dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transform
        )
    elif dataset_name.lower() == 'monet':
        dataset = MonetDataset(root_dir, transform=transform)
    else:
        raise ValueError("Dataset not recognized.")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    return dataloader
