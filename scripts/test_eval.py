import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

def test_eval(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())
            correct += (torch.sigmoid(outputs) > 0.5).eq(targets.bool()).all(dim=1).sum().item()
            total += targets.size(0)

    test_loss = test_loss / total
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")