import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Model(nn.Module):
    """

    Pour cette tache, on utilisera un perceptron multicouche (donnée MNIST) avec 2 couches cachées de 200 unités chacunes et une focntion d'activation ReLu

    Un CNN avec 2 couches de convolution de 5*5 (32, 64 cannaux) suivie d'un poolmax de 2*2, une couche entierrement connecte de 512 unite d'activation ReLu, une couche de sortie softmax

    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512) # Couche entierement connectée
        self.fc2 = nn.Linear(512, 10) # On aura en sortie 10 classes

    def forward(self, inputs: torch.Tensor): 
        x = inputs.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = x.view(-1, 7*7*64)
        x = nn.Flatten()(x) # If decomment line up, comment it. 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train(model, local_epoch: int, traindataloader, optimizer, device: str = "cpu"):
    """Fonction d'entrainement du modèle"""

    loss_fn = F.cross_entropy

    for epoch in range(local_epoch): 
        for data, label in traindataloader: 
            data, label = data.to(device), label.to(device)
            predictions =  model(data)
            loss = loss_fn(predictions, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model.state_dict()



def test():
    """Fonction de test du modèle"""
