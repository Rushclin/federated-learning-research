import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision.datasets import ImageFolder

class GetDataSet: 
    def __init__(self, path: str = r"./data") -> None:
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
        
        self.classes_size = len(image_datasets['train'].classes)
    

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
    print("Taille des classes  :", dataset.classes_size)

    show_random_images(dataset.image_datasets['train'])
