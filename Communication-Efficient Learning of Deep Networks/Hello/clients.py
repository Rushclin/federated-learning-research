import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset import GetDataset 
import torch.nn.functional as F
# from dset import prepare_dataset
import matplotlib.pyplot as plt

class Client(object):
    def __init__(self, train_dataset, device):
        self.train_ds = train_dataset
        self.device = device
        self.train_dl = None
        self.local_parameters = None

    def local_update(self, net, local_epoch, local_batch_size, optimizer, global_parameters):

        loss_fun = F.cross_entropy

        net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=local_batch_size, shuffle=True)

        for epoch in range(local_epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.device), label.to(self.device)
                preds = net(data)
                loss = loss_fun(preds, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, num_of_clients: int, device: str = 'cpu', is_iid: bool = True):
        self.is_iid = is_iid
        self.num_of_clients = num_of_clients
        self.device = device
        self.clients_set = {}

        self.test_data_loader = None

        self.dataset_balance_allocation()


    def dataset_balance_allocation(self):

        # trainloaders, testloader = prepare_dataset(batch_size=self.num_of_clients, num_partition=10)

        # self.test_data_loader = testloader

        # print("Taill ==>", len(trainloaders))

        # for i, trainloader in enumerate(trainloaders):
        #     client_data = next(iter(trainloader))
        #     client_label = torch.randint(0, 10, (client_data.size(0),))
        #     someone = Client(TensorDataset(client_data, client_label), self.device)
        #     self.clients_set['client{}'.format(i)] = someone

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
    MyClients = ClientsGroup(is_iid=True, num_of_clients=100)
    print(MyClients.clients_set['client10'].train_ds[:5])
    print(MyClients.clients_set['client11'].train_ds[400:405])

    batch = next(iter(MyClients.test_data_loader))
    images, labels = next(iter(MyClients.test_data_loader))

    print("Images ==>",images[0][0], "Labels ==>", labels[0])

    # Assuming images is a 4D tensor (batch_size, channels, height, width)
    # You can display the first image in the batch
    plt.imshow(images[0])
    plt.title(f"Label: {labels[0]}")
    plt.show()