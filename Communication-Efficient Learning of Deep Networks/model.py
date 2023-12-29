import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2) # Déclaration d'une couche de convolution avec 1 canal d'entrée, 32 canaux de sortie, une taille de noyau de 5x5, un pas de 1 et un bourrage (padding) de 2.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # Déclaration d'une couche de pooling (max pooling) avec une fenêtre de 2x2 et un pas de 2.
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        """
            Ici, on définie comment la méthode sera propagée
        """
        x = inputs.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
