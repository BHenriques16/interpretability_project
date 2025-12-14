import torch.nn as nn
import torchvision.models as models

class PretrainedModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PretrainedModel, self).__init__()
        # Loads Resnet-18
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
