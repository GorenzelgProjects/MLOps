import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np

# Define a custom dataset class
class CustomMNISTDataset(Dataset):
    def __init__(self, images_file, targets_file, root_dir, transform=None):
        self.images = torch.load(os.path.join(root_dir, images_file))
        self.targets = torch.load(os.path.join(root_dir, targets_file))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, target


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset

    # Path to the dataset
    root_dir = './corruptmnist'

    # Transforms can be added to the dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])


    # Create the custom datasets
    train_dataset = CustomMNISTDataset(images_file='train_images_0.pt',
                                    targets_file='train_target_0.pt',
                                    root_dir=root_dir,
                                    transform=transform)

    test_dataset = CustomMNISTDataset(images_file='test_images.pt',
                                    targets_file='test_target.pt',
                                    root_dir=root_dir,
                                    transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Return the dataloaders
    return train_loader, test_loader

if __name__ == '__main__':
    train, test = mnist()
    #print(train[0].shape)
    #print(train[1].shape)
    #print(test[0].shape)
    #print(test[1].shape)