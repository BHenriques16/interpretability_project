import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Definir dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformações: Resize para 128x128, converter para tensor e normalizar
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Carregar dataset CelebA completo
celeba_dataset = datasets.CelebA(root='./data', split='all', transform=transform, download=True)

# Divisão treino/validação/teste: 70%, 15%, 15%
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

# Confirmação de tamanhos
print(f'Tamanho treino: {len(train_data)}')
print(f'Tamanho validação: {len(val_data)}')
print(f'Tamanho teste: {len(test_data)}')