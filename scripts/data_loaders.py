import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Resize to 128x128, convert to tensor, and normalize and data augmentation for better generalization
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # random horizontal flip
    transforms.RandomRotation(15),      # random rotation up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # random brightness and contrast adjustment
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load complete CelebA dataset
celeba_dataset = datasets.CelebA(root='./data', split='all', transform=train_transform, download=True)

# Dataset split sizes train/val/test: 70%, 15%, 15%
total_size = len(celeba_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(celeba_dataset, [train_size, val_size, test_size])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Confirm sizes
print(f'Train size: {len(train_data)}')
print(f'Validation size: {len(val_data)}')
print(f'Test size: {len(test_data)}')