import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision.datasets import ImageFolder

class GetDataSet: 
    def __init__(self, path: str = r"./data") -> None:
        self.train_data = None;
        self.train_label = None; 
        self.train_data_size = None

        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self.label_images_str = None
        self.classes_size = None

        self.path = path

        self.get_dataset()


    def get_dataset(self): 

        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Grayscale(num_output_channels=3),  # Convertir en RGB - ou du moins laisser les images telles qu'elles sont
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform = transforms.Compose([ 
            transforms.PILToTensor() 
        ]) 

        plant_village_dataset = ImageFolder(
            root=self.path,
            transform=transform # TODO: Remove to add preprocessing
        )

        self.label_images_str = plant_village_dataset.classes

        data_size = len(plant_village_dataset)
        train_size = int(0.8 * data_size)
        test_size = data_size - train_size

        self.classes_size = len(plant_village_dataset.classes)

        self.train_data, self.test_data = torch.utils.data.random_split(plant_village_dataset, [train_size, test_size])

        self.train_label = [label for _, label in self.train_data]
        self.test_label = [label for _, label in self.test_data]

        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)


def map_label_to_class(dataset, label, list_label_str):
    for _, l in dataset: 
        if(l == label):
            return list_label_str[l]
    

def show_random_images(dataset, list_label_str, num_images=5): 
    random_indices = np.random.choice(len(dataset), num_images, replace=False)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i, idx in enumerate(random_indices):
        image, label = dataset[idx]
        axes[i].imshow(image[0]) 
        axes[i].set_title(f"{map_label_to_class(dataset, label, list_label_str)}")
        axes[i].axis('off')

    plt.show()


if __name__ == "__main__": 
    
    dataset = GetDataSet()
    print("Taille de l'ensemble d'entraînement :", dataset.train_data_size)
    print("Taille de l'ensemble de test :", dataset.test_data_size)

    show_random_images(dataset.train_data, dataset.label_images_str)
