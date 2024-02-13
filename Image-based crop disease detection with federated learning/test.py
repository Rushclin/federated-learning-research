import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim



class Models(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, device='cuda'):
        super(Models, self).__init__()

        self.model = models.resnet50(pretrained=pretrained).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        ).to(device)

    def forward(self, x):
        return self.model(x)




def main():
        

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'validation':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        'train': 
        datasets.ImageFolder('data/train', data_transforms['train']),
        'validation': 
        datasets.ImageFolder('data/validation', data_transforms['validation'])
    }

    dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=32,
                                    shuffle=True, num_workers=4),
        'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=32,
                                    shuffle=False, num_workers=4)
    }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = models.resnet50(pretrained=True).to(device)
        
    # for param in model.parameters():
    #     param.requires_grad = False   
        
    # model.fc = nn.Sequential(
    #             nn.Linear(2048, 128),
    #             nn.ReLU(inplace=True),
    #             nn.Linear(128, 5)).to(device)

    model = Models(num_classes=5, pretrained=True, device=device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.fc.parameters())


    def train_model(model, criterion, optimizer, num_epochs=3):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    print("L'entree est ===>",inputs)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.detach() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.float() / len(image_datasets[phase])

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss.item(),
                                                            epoch_acc.item()))
        return model



    model_trained = train_model(model, criterion, optimizer, num_epochs=3)



    # torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')

    # model = models.resnet50(pretrained=False).to(device)
    # model.fc = nn.Sequential(
    #             nn.Linear(2048, 128),
    #             nn.ReLU(inplace=True),
    #             nn.Linear(128, 2)).to(device)
    # model.load_state_dict(torch.load('models/pytorch/weights.h5'))



if __name__ == '__main__':
    main()

