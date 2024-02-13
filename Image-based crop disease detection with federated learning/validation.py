import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


def main(): 
    print("Bonjour")

    model = models.resnet50(pretrainded = True).to('cpu');

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )

    model.load_state_dict(torch.load('/weigths'))

if __name__ == "__main__":
    main()