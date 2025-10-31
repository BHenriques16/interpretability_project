import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from data_loaders import train_loader, val_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN(num_class=40).to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy para multi-label
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_val_loss = float('inf')  # Para guardar o melhor modelo

# Treino
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validação
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    val_epoch_loss = val_loss / len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

    # guardar o melhor modelo
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), 'best_model_celeba.pth')
        print(f'Melhor modelo salvo com loss de validação: {best_val_loss:.4f}')