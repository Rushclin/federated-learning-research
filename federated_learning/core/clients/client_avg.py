# Federated Learning
# Copyright (C) 2024  Takam Rushclin

# Client AVG

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from utils.dataset import GetDataSet

class Client:
    def __init__(self, client_id: int, train_dataset, device: str = "cpu"):
        self.client_id = client_id

        self.train_dataset = train_dataset
        self.train_dataloader = None
        self.device = device

        self.local_parameters = None

    def clientUpdate(self, model, optimizer, global_parameters, num_epochs: int = 1, batch_size: int = 10, device: str = "cpu"):
        loss_fn = F.cross_entropy

        # Chargement des parametres globaux du modele
        model.load_state_dict(global_parameters, strict=True)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size, shuffle=True)

        for epoch in range(num_epochs):

            model.train()

            for data, label in self.train_dataloader:

                data, label = data.to(device), label.to(device)

                pred = model(data)

                loss = loss_fn(pred, label)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

        return model.state_dict()


class ClientGroup:
    def __init__(self, 
                 num_of_clients: int, 
                 batch_size: int,  
                 input_folder: str = r"./../../data",
                 output_folder: str = r"./../../dataset", 
                 image_size: int = 28, 
                 non_iid: bool = True, 
                 device: str = "cpu"):

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_size = image_size

        self.num_of_clients = num_of_clients
        self.non_iid = non_iid
        self.batch_size = batch_size

        self.device = device

        self.test_data_loader = None

        self.clients_set = {}

        self.balance_dataset()

    def balance_dataset(self):

        dataset = GetDataSet(input_folder=self.input_folder,
                             output_folder=self.output_folder, image_size=self.image_size)

        train_data = dataset.image_datasets['train']

        subset_size = len(train_data) // self.num_of_clients

        self.test_data_loader = DataLoader(
            dataset=dataset.image_datasets['validation'], batch_size=self.batch_size, shuffle=False)

        for i in range(self.num_of_clients):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size

            subset = Subset(train_data, range(start_idx, end_idx))

            someone = Client(i, subset, self.device)

            self.clients_set[f'client{i}'] = someone


if __name__ == "__main__":

    print("Test du module Client et ClientGroup")

    clientGroup = ClientGroup(3, 10)
    print(clientGroup.clients_set)
