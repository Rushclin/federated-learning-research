import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader


def get_mnist(data_patch: str = './data'): 

    """
        Fonction qui doit nous permèttre de télécharger le dataset (trainset et testset).
        On applique une transformation ToTensor() sur ces données images
    """
    
    trainset = MNIST(
        root=data_patch,
        train=True, 
        download=True,
        transform=ToTensor()
    )

    testset = MNIST(
        root=data_patch,
        download=True, 
        train=False,
        transform=ToTensor()
    )

    return trainset, testset

def prepare_dataset(
        num_partition: int, 
        batch_size: int, 
        val_ratio: float = 0.1):

    trainset, testset = get_mnist()

    num_images = len(trainset) // num_partition

    partition_len = [num_images] * num_partition

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    trainloaders = []
    valloaders = []

    for trainset_ in trainsets: 
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)) 
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=128)


    return trainloaders, valloaders, testloader