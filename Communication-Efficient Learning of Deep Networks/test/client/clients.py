import torch
import numpy as np 

from torch.utils.data import DataLoader, TensorDataset

from models.model import Model, train, test
from datasets.dataset import prepare_dataset


class Client(object): 
    def __init__(self, trainloader, device: str = 'cpu'):
        self.trainloader = trainloader
        self.device = device
        self.local_params = None
        self.model = Model()

    def local_update(
            self, 
            local_epoch: int, 
            local_batch_size: int, 
            # loss_fn, 
            optimizer, 
            global_params
        ):

        self.model.load_state_dict(global_params, strict=True)
       
        traindataloader = DataLoader(
            self.trainloader, 
            batch_size=local_batch_size, 
            shuffle=True
        )

        # for epoch in range(local_epoch): 
        #     for data, label in traindataloader: 
        #         data, label = data.to(self.device), label.to(self.device)
        #         predictions =  self.model(data)
        #         loss = loss_fn(predictions, label)
        #         loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()

        # return self.model.state_dict()

        return train(self.model, local_epoch, traindataloader, optimizer)


class ClientsGroup(object):
    def __init__(self, num_client: int, is_iid: bool = True, device: str = 'cpu') :
        self.device = device
        self.clients = []
        self.num_client = num_client
        self.is_iid = is_iid

        self.traindataloader = None


    def dataset_balance_allocation(self): 

        trainloaders, valloaders, testloader = prepare_dataset(batch_size=100, num_partition=10)

        shard_size = len(trainloaders) // self.num_of_clients // 2
        shards_id = np.random.permutation(len(trainloaders) // shard_size)

        # for i in range(self.num_client): 
            # somone = Client(TensorDataset(torch.tensor()))    
            # shards_id1 = shards_id[i * 2]
            # shards_id2 = shards_id[i * 2 + 1]
            # data_shards1 = trainloaders[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # data_shards2 = trainloaders[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # local_label = np.argmax(local_label, axis=1)
            # someone = Client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.device)