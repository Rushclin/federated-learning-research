# import torch
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt

# class GetDataset(object):
#     def __init__(self, is_iid: bool = True) -> None:
#         self.train_data = None
#         self.train_label = None
#         self.train_data_size = None

#         self.test_data = None
#         self.test_label = None
#         self.test_data_size = None

#         self._index_in_train_epoch = 0

#         self.prepare_dataset(is_iid)

#     def prepare_dataset(self, is_iid: bool, root: str = './datas'):
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#         train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
#         test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

#         self.train_data_size = len(train_dataset)
#         self.test_data_size = len(test_dataset)

#         if is_iid:
#             indices = torch.randperm(self.train_data_size)
#             self.train_data = train_dataset.data[indices].view(self.train_data_size, -1).float() / 255.0
#             # self.train_label = torch.eye(10)[train_dataset.targets[indices]]
#             self.train_label = train_dataset.targets[indices].unsqueeze(1)
#         else:
#             sorted_indices = torch.argsort(train_dataset.targets)
#             self.train_data = train_dataset.data[sorted_indices].view(self.train_data_size, -1).float() / 255.0
#             # self.train_label = torch.eye(10)[train_dataset.targets[sorted_indices]]
#             self.train_label = train_dataset.targets[sorted_indices].unsqueeze(1)

#         self.test_data = test_dataset.data.view(self.test_data_size, -1).float() / 255.0
#         # self.test_label = torch.eye(10)[test_dataset.targets]
#         self.test_label = test_dataset.targets.unsqueeze(1)

# if __name__ == "__main__":
#     mnist_dataset = GetDataset(is_iid=True)

#     if type(mnist_dataset.train_data) is torch.Tensor and type(mnist_dataset.test_data) is torch.Tensor and \
#             type(mnist_dataset.train_label) is torch.Tensor and type(mnist_dataset.test_label) is torch.Tensor:
#         print('Le type de donnée est bien un tensor PyTorch')
#     else:
#         print('Le type de donnée n\'est pas un tensor PyTorch')

#     print('La forme du dataset d\'entrainnement est  {}'.format(mnist_dataset.train_data.shape))
#     print('La forme du dataset de test est {}'.format(mnist_dataset.test_data.shape))
#     print(mnist_dataset.train_label[:100], mnist_dataset.train_label[11000:11100])































import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class GetDataset(object):
    def __init__(self, is_iid: bool = True) -> None:
        self.train_data = None
        self.train_label = None
        self.train_data_size = None

        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        self.prepare_dataset(is_iid)

    def prepare_dataset(self, is_iid: bool, root: str = './datas'):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        self.train_data_size = len(train_dataset)
        self.test_data_size = len(test_dataset)

        # print(self.train_data_size)
        # print(self.test_data_size)

        if is_iid:
            indices = torch.randperm(self.train_data_size)
            self.train_data = train_dataset.data[indices].view(self.train_data_size, -1).float() / 255.0
            self.train_label = train_dataset.targets[indices].unsqueeze(1)
        else:
            sorted_indices = torch.argsort(train_dataset.targets)
            self.train_data = train_dataset.data[sorted_indices].view(self.train_data_size, -1).float() / 255.0
            self.train_label = train_dataset.targets[sorted_indices].unsqueeze(1)

        self.test_data = test_dataset.data.view(self.test_data_size, -1).float() / 255.0
        self.test_label = test_dataset.targets.unsqueeze(1)

if __name__ == "__main__":
    mnist_dataset = GetDataset(is_iid=True)

    if type(mnist_dataset.train_data) is torch.Tensor and type(mnist_dataset.test_data) is torch.Tensor and \
            type(mnist_dataset.train_label) is torch.Tensor and type(mnist_dataset.test_label) is torch.Tensor:
        print('Le type de donnée est bien un tensor PyTorch')
    else:
        print('Le type de donnée n\'est pas un tensor PyTorch')

    print('La forme du dataset d\'entrainnement est  {}'.format(mnist_dataset.train_data.shape))
    print('La forme du dataset de test est {}'.format(mnist_dataset.test_data.shape))
    print(mnist_dataset.train_label[:100], mnist_dataset.train_label[11000:11100])

    print(mnist_dataset.test_label[969])
    print(mnist_dataset.test_data[0])
