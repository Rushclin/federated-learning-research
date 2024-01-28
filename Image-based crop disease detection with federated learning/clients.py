import torch

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, TensorDataset

import torch.nn.functional as F

from dataset import GetDataSet


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
            for i, inp in enumerate(self.train_dataloader):
                data, label = inp
                data, label = data.to(device), label.to(device)
                pred = model(data)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model.state_dict()


class ClientGroup:
    def __init__(self, num_of_clients: int, batch_size: int, path: str = r"./data", non_iid: bool = True, device: str = "cpu"):
        self.path = path
        self.num_of_clients = num_of_clients
        self.non_iid = non_iid
        self.batch_size = batch_size

        self.device = device

        self.test_data_loader = None

        self.clients_set = {}

        self.balance_dataset()

    def balance_dataset(self):
        """
            Répartition des données pour les différents clients 
        """

        dataset = GetDataSet(path=self.path)

        self.test_data_loader = DataLoader(
            dataset.test_data, batch_size=self.batch_size, shuffle=False)

        train_data = dataset.train_data

        train_label_np = np.array(dataset.train_label)
        train_data_extract = []

        for data, _ in enumerate(train_data):
            train_data_extract.append(data)

        shard_size = dataset.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(
            dataset.train_data_size // shard_size)

        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]

            data_shards1 = train_data_extract[shards_id1 *
                                              shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data_extract[shards_id2 *
                                              shard_size: shards_id2 * shard_size + shard_size]

            label_shards1 = train_label_np[shards_id1 *
                                           shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label_np[shards_id2 *
                                           shard_size: shards_id2 * shard_size + shard_size]

            local_data = torch.tensor(np.vstack((data_shards1, data_shards2)))
            local_label = torch.tensor(
                np.vstack((label_shards1, label_shards2)))
            local_label = torch.argmax(local_label, axis=1)

            local_dataset = TensorDataset(local_data, local_label)
            someone = Client(i, local_dataset, self.device)

            self.clients_set['client{}'.format(i)] = someone


if __name__ == "__main__":

    print("Test du module Client et ClientGroup")

    clientGroup = ClientGroup(3, 10)
    print(clientGroup.clients_set)
