import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformações: converter para tensor e normalizar
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Carregar dataset CelebA (certifica-te de ter o arquivo de anotações e imagens)
celeba_dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)

# Dividir em treino e validação (80/20)
total_size = len(celeba_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_data, val_data = random_split(celeba_dataset, [train_size, val_size])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Confirma os tamanhos
print(f'Tamanho treino: {len(train_data)}')
print(f'Tamanho validação: {len(val_data)}')
