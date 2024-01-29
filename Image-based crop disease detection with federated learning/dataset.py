import torch

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

class GetDataSet: 
    def __init__(self, path: str = r"./data") -> None:
        # self.train_data = None;
        # self.train_label = None; 
        # self.train_data_size = None

        # self.test_data = None
        # self.test_label = None
        # self.test_data_size = None

        # self.label_images_str = None
        self.classes_size = None

        self.image_datasets = None

        self.path = path

        self.get_dataset()


    def get_dataset(self): 

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
            ImageFolder(f'{self.path}/train', data_transforms['train']),
            'validation': 
            ImageFolder(f'{self.path}/validation', data_transforms['validation'])
        }

        self.image_datasets = image_datasets
        
        self.classes_size = len(image_datasets)



        # preprocessing = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     # transforms.CenterCrop(224),
        #     # transforms.Grayscale(num_output_channels=3),  # Convertir en RGB - ou du moins laisser les images telles qu'elles sont
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])

        # transform = transforms.Compose([ 
        #     transforms.PILToTensor() 
        # ]) 

        # plant_village_dataset = ImageFolder(
        #     root=self.path,
        #     transform=preprocessing 
        # )

        # self.label_images_str = plant_village_dataset.classes

        # data_size = len(plant_village_dataset)
        # train_size = int(0.8 * data_size)
        # test_size = data_size - train_size

        # self.classes_size = len(plant_village_dataset.classes)

        # self.train_data, self.test_data = random_split(plant_village_dataset, [train_size, test_size])
    
        # self.train_label = [label for _, label in self.train_data]
        # self.test_label = [label for _, label in self.test_data]

        # self.train_data_size = len(self.train_data)
        # self.test_data_size = len(self.test_data)
    

def show_random_images(dataset, num_images=5): 
    random_indices = np.random.choice(len(dataset), num_images, replace=False)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i, idx in enumerate(random_indices):
        image, label = dataset[idx]
        axes[i].imshow(image[0]) 
        axes[i].set_title(f"{dataset.classes[i]}")
        axes[i].axis('off')

    plt.show()


if __name__ == "__main__": 
    
    dataset = GetDataSet()
    print("Taille de l'ensemble d'entraînement :", len(dataset.image_datasets['train']))
    print("Taille de l'ensemble de test :", len(dataset.image_datasets['validation']))

    show_random_images(dataset.image_datasets['train'])
