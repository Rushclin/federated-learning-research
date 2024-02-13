# Federated Learning
# Copyright (C) 2024  Takam Rushclin

# Model CNN

import torch.nn as nn
import torch.nn.functional as F


class Model_2NN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN_Model(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        # Couche qui doit permettre d'éteindre certains neurones afin d'éviter le sur-apprentissage
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    num_channels, num_classes = 3, 10 # Pour des images avec couleurs RGB (3) et de 10 classes
    model = CNN_Model(num_channels, num_classes)

    print(model)
