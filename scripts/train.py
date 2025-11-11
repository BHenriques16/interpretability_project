import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from model import CNN
from data_transform import create_data_transforms
from test_eval import test_eval
from model import PretrainedModel

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training and validation functions
def train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == targets.bool()).all(dim=1).sum().item()
        total += targets.size(0)
        loop.set_postfix(loss=running_loss/total)
    return running_loss/total, 100.*correct/total

def validate(model, val_loader, criterion, val_dataset, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == targets.bool()).all(dim=1).sum().item()
            total += targets.size(0)
    return val_loss / len(val_dataset), 100.*correct/total

# Main training loop with early stopping and model saving
def train(model, train_loader, val_loader, optimizer, criterion, max_epochs, device, model_save_path, patience=5):
    best_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    early_stopping = EarlyStopping(patience=patience)

    total_start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device)
        val_loss, val_acc = validate(model, val_loader, criterion, val_loader.dataset, device)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{max_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f} seconds")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved (Validation Loss: {best_loss:.4f})")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Training complete! Triggered early stopping. Total time: {total_duration:.2f} seconds")

# MAIN
if __name__ == "__main__":
    batch_size = 16
    img_size = 128
    max_epochs = 50
    lr = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_save_path = "models/best_model.pth"
    train_transforms, val_transforms = create_data_transforms(img_size)

    # CelebA dataset (uses split all and manually divides with random_split)
    celeba_dataset = datasets.CelebA(root="./data", split="all", target_type="attr", download=True, transform=train_transforms)
    total_size = len(celeba_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_data, val_data, test_data = random_split(celeba_dataset, [train_size, val_size, test_size])

    # Show dataset sizes
    print(f'Train size: {len(train_data)}')
    print(f'Validation size: {len(val_data)}')
    print(f'Test size: {len(test_data)}')

    # Ajust transform to val_data for validation without augmentation
    val_data.dataset.transform = val_transforms
    test_data.dataset.transform = val_transforms

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = 40  # CelebA has 40 classes

    model = PretrainedModel(num_classes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(model, train_loader, val_loader, optimizer, criterion, max_epochs, device, model_save_path, patience=3)

# Load the best model and evaluate on test set
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)

    test_eval(model, test_loader, criterion, device)