import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset import GetDataset  # Assurez-vous d'importer correctement le module

class Client(object):
    def __init__(self, train_dataset, device):
        self.train_ds = train_dataset
        self.device = device
        self.train_dl = None
        self.local_parameters = None

    def local_update(self, net, local_epoch, local_batch_size, loss_fun, opti, global_parameters):
        net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=local_batch_size, shuffle=True)
        for epoch in range(local_epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.device), label.to(self.device)
                preds = net(data)
                loss = loss_fun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, is_iid: bool, num_of_clients: int, device: str = 'cpu'):
        self.is_iid = is_iid
        self.num_of_clients = num_of_clients
        self.device = device
        self.clients_set = {}

        self.test_data_loader = None

        self.dataset_balance_allocation()

    def dataset_balance_allocation(self):
        mnist_dataset = GetDataset(is_iid=self.is_iid)

        test_data = torch.tensor(mnist_dataset.test_data)
        test_label = torch.tensor(mnist_dataset.test_label)

        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        
        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label

        shard_size = mnist_dataset.train_data_size // self.num_of_clients // 2
        shards_id = torch.randperm(mnist_dataset.train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data = torch.cat([data_shards1, data_shards2])
            # local_label = torch.argmax(torch.cat([label_shards1, label_shards2]))

            local_label = torch.argmax(torch.cat([label_shards1, label_shards2]), dim=1)

            someone = Client(TensorDataset(local_data, local_label), self.device)
            self.clients_set['client{}'.format(i)] = someone

if __name__ == "__main__":
    MyClients = ClientsGroup(is_iid=True, num_of_clients=100, device='cpu')
    print(MyClients.clients_set['client10'].train_ds[:5])
    print(MyClients.clients_set['client11'].train_ds[400:405])
