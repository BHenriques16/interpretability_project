import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import time
from data_loaders import train_loader, val_loader
from scripts.model import CNN

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Initialize model, loss function, and optimizer
model = CNN(num_class=40).to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy para multi-label
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
num_epochs = 50  
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Track total training time
start_training = time.time()

# Training process (train on batches)
for epoch in range(num_epochs):
    start_epoch = time.time()
    
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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    val_epoch_loss = val_loss / len(val_loader.dataset)

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Epoch Time: {epoch_time:.2f}s')

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), 'best_model_celeba.pth')
        print(f'Best model saved with validation loss: {best_val_loss:.4f}')
        epochs_no_improve = 0  # reset patience
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping activated after {epoch+1} epochs without improvement.')
            break

# Total training time
end_training = time.time()
total_training_time = end_training - start_training
print(f'Total training time: {total_training_time:.2f}s')