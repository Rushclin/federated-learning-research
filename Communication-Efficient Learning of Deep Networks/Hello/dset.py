import torch 
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import MNIST 
from torch.utils.data import DataLoader, random_split

class GetDataset(object): 
    def __init__(self, is_iid: bool = True):
        self.train_data = None
        self.train_label = None
        
        self.test_data = None 
        self.test_label = None 

def get_mnist(data_path: str = './datas'):

    tr = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    trainset = MNIST(
        root=data_path, 
        train=True, 
        transform=tr, 
        download=True
    )

    testset = MNIST(
        root=data_path,
        download=True, 
        transform=tr, 
        train=False
    )

    return trainset, testset


def prepare_dataset(batch_size: int,
                    num_partition: int,
                    val_ratio: float = 0.1,
                    is_iid: bool = True):
    trainset, testset = get_mnist()

    num_images = len(trainset) // num_partition

    partition_len = [num_images] * num_partition

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    trainloaders = []
    # valloaders = []

    # for trainset_ in trainsets: 
    #     num_total = len(trainset_)
    #     num_val = int(val_ratio * num_total)
    #     num_train = num_total - num_val

    #     for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)) 
        
    #     trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
    #     valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=128)
    trainloaders = DataLoader(trainset, batch_size=128)

    return trainloaders, testloader


if __name__ == "__main__": 
    trainloaders, testloader = prepare_dataset(batch_size=10, num_partition = 100)

    images, labels = next(iter(trainloaders[0]))
    
    plt.imshow(images[0][0])  # La première image du premier lot
    plt.title(f"Label: {labels[0]}")
    plt.show()
