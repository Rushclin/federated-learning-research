import torch
from torchvision import datasets, transforms

class GetDataSet(object):
    """
        GetDataSet 

        Prend un paramètre booléen is_iid qui indique si les données doivent être IID (Indépendantes et Identiquement Distribuées) ou non.

        Cette classe est conçue pour fournir des données MNIST prêtes à l'emploi, \n
        avec des options pour spécifier si les données doivent être IID ou non. \n 
        Elle applique également des transformations telles que la normalisation pour préparer les données pour l'entraînement d'un modèle.
    """
    def __init__(self, is_iid):
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self.get_dataset(is_iid)

    def get_dataset(self, is_iid):
        """
            Téléchargement du dataset MNIST
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # Les images MNIST sont transformées en tenseurs PyTorch, puis normalisées.

        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.train_data_size = len(trainset)
        self.test_data_size = len(testset)

        self.train_data = trainset.data.view(self.train_data_size, -1).float() / 255.0 # Mise en forme des données d'entraînement en un tableau 2D et normalisation des valeurs des pixels entre 0 et 1.
        self.train_label = torch.eye(10)[trainset.targets] #  Encodage one-hot des étiquettes d'entraînement en utilisant torch.eye(10).

        self.test_data = testset.data.view(self.test_data_size, -1).float() / 255.0
        self.test_label = torch.eye(10)[testset.targets]

        if not is_iid:
            sorted_indices = torch.argsort(torch.argmax(self.train_label, dim=1))
            self.train_data = self.train_data[sorted_indices]
            self.train_label = self.train_label[sorted_indices]