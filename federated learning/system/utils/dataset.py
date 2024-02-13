# Federated Learning
# Copyright (C) 2024  Takam Rushclin

# Class GetDataset

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder

from split_dataset import split_dataset


class GetDataSet:
    def __init__(self, input_folder: str = r"./../../data", output_folder: str = r"./../../dataset", image_size : int= 28):
        """
        Input : \n 
             input_folder : ça représente le chemin du dataset \n
             output_folder : ici c'est le chemin du datastet apres avoir effectué le split (80% pour l'entrainement et 20% pour la validation) \n 
             C'est ce chemin qu'on doit utiliser pour le travail
             image_size : représente la taille des images, on suppose que les images ont en hauteur et en largeur la même taille
        """
        
        
        self.classes_size = None

        self.image_datasets = None
        self.input_folder = input_folder
        self.output_folder = output_folder

        self.image_size = image_size

        # Get dataset
        self.get_datastet()

    def get_datastet(self):

        split_dataset(input_folder=self.input_folder,
                      output_folder=self.output_folder, train_ratio=0.8)

        normalise = transforms.Normalize((0.5), (0.5))

        data_transform = {
            'train':
            transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalise
            ]),
            'validation':
            transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalise
            ])
        }

        image_datasets = {
            'train':
            ImageFolder(f'{self.output_folder}/train',
                        data_transform['train']),
            'validation':
            ImageFolder(f'{self.output_folder}/validation',
                        data_transform['validation'])
        }

        self.image_datasets = image_datasets

        self.classes_size = len(image_datasets['train'].classes)


def show_random_images(dataset, num_images=5):
    random_indices = np.random.choice(len(dataset), num_images, replace=False)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i, idx in enumerate(random_indices):
        image, label = dataset[idx]
        axes[i].imshow(image[0])
        axes[i].set_title(f"{label}")
        axes[i].axis('off')

    plt.show()


if __name__ == "__main__": 
    
    dataset = GetDataSet()
    print("Taille de l'ensemble d'entraînement :", len(dataset.image_datasets['train']))
    print("Taille de l'ensemble de test :", len(dataset.image_datasets['validation']))
    print("Taille des classes  :", dataset.classes_size)

    show_random_images(dataset.image_datasets['train'])
