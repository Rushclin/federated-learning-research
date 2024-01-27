import torch

import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset, TensorDataset

import torch.nn.functional as F

from dataset import GetDataSet


class Client:
    def __init__(self, client_id, train_dataset, device: str = "cpu"):
        self.client_id = client_id

        self.train_dataset = train_dataset
        self.train_dataloader = None
        self.device = device

        self.local_parameters = None

    def update_model(self, model, optimizer, global_parameters, num_epochs: int = 1, batch_size: int = 10, device: str = "cpu"):
        loss_fn = F.cross_entropy

        # Chargement des parametres globaux du modele
        model.load_state_dict(global_parameters, strict=True)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for data, label in self.train_dataloader:
                data, label = data.to(device), label.to(device)
                pred = model(data)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return model.state_dict()


class ClientGroup:
    def __init__(self, num_clients: int, batch_size: int, path: str = r"./../data", non_iid: bool = True):
        self.path = path
        self.num_clients = num_clients
        self.non_iid = non_iid
        self.batch_size = batch_size

        self.test_data_loader = None

        self.clients_set = {}

        self.balance_dataset()

    def balance_dataset(self):
        """
            Repartition des données pour les différents clients
        """

        plant_village_dataset = GetDataSet(path=self.path)

        label_to_index = {label: idx for idx, label in enumerate(plant_village_dataset.test_label)}
        test_label_numeric = [label_to_index[label] for label in plant_village_dataset.test_label]




        test_dataset = TensorDataset(
            torch.tensor(plant_village_dataset.test_data),
            torch.tensor(test_label_numeric)
        )

        print(test_dataset)

        self.test_data_loader = DataLoader(
            TensorDataset(
                plant_village_dataset.test_data, plant_village_dataset.test_label
            ),
            batch_size=self.batch_size, shuffle=False
        )

        print("Loader ==> ",self.test_data_loader)

        train_data = plant_village_dataset.train_data
        train_label = plant_village_dataset.train_label

        shard_size = plant_village_dataset.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(
            plant_village_dataset.train_data_size // shard_size)

        for i in range(self.num_clients):
            print("II", i)

            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 *
                                      shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 *
                                      shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 *
                                        shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 *
                                        shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack(
                (data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)

            someone = Client(TensorDataset(torch.tensor(
                local_data), torch.tensor(local_label)), self.device)

            # self.clients_set['client{}'.format(i)] = someone
            print(someone)


if __name__ == "__main__":
    print("Test du module ClientGroup et du module Client")

    client_group = ClientGroup(
        num_clients=1, batch_size=10, path=r"./../data", non_iid=False)
    print(client_group.clients_set)
