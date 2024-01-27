import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class GetDataSet(object): 
    def __init__(self, path: str = r"./../data"):
        self.train_data = None;
        self.train_label = None; 
        self.train_data_size = None

        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self.path = path

        self.get_dataset()


    def get_dataset(self):
        
        # Transformations à appliquer sur les images
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),  # Convertir en niveau de gris
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Seulement une seule chaîne de couleur maintenant
        ])

        plant_village_dataset = ImageFolder(
            root=self.path,
            transform=preprocessing
        )

        train_size = int(0.8 * len(plant_village_dataset))
        test_size = len(plant_village_dataset) - train_size

        self.test_data_size = test_size
        self.train_data_size = train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            plant_village_dataset, [train_size, test_size])
        
        self.train_data = train_dataset.dataset
        self.train_label = train_dataset.dataset.classes

        self.test_data = test_dataset.dataset
        self.test_label = test_dataset.dataset.classes


if __name__ == "__main__": 
    
    dataset = GetDataSet()
    print("Taille de l'ensemble d'entraînement :", dataset.train_data_size)
    print("Taille de l'ensemble de test :", dataset.test_data_size)