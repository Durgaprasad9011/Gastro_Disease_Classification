import torch
import torch.nn as nn
from torchvision import models

class PrototypicalNet(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(PrototypicalNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.fc = nn.Linear(in_features, feature_dim)
        self.prototypes = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
