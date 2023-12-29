import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from datasets import GetDataSet
import torch.nn.functional as F


class Client(object):
    """
        La classe qui doit créer un client avec son dataset d'entraînement

        Prend en entrée: 
            - le train_dataset qui est le dataset sur lequel le client doit faire son entraînnement
            - le device, par defaut le CPU
    """

    def __init__(self, train_dataset, device: str = 'cpu'):
        self.train_ds = train_dataset
        self.device = device
        self.train_dl = None
        self.local_parameters = None

    def local_update(self, local_epoch, local_batch_size, net, optimizer, global_parameters):
        """
            local_update


            Méthode pour effectuer la mise à jour locale du modèle sur le client. \n
            Il charge les paramètres globaux (global_parameters),  \n
            utilise un DataLoader pour itérer sur les données locales,  \n 
            calcule la perte, effectue la rétropropagation  
            et la mise à jour des paramètres avec l'optimiseur. 

            Prend en entrée: 
                - local_epoch : Nombre d'époques (itérations complètes sur les données locales) à effectuer lors de la mise à jour locale. Une époque correspond à une passe complète à travers toutes les données locales.
                - local_batch_size
                - net : Modèle PyTorch qui doit être mis à jour localement. C'est le modèle initialisé au niveau global et chargé avec les paramètres globaux avant la mise à jour locale.
                - optimizer : Optimiseur PyTorch utilisé pour mettre à jour les poids du modèle pendant l'entraînement. Il s'agit de l'optimiseur qui sera utilisé pour appliquer les mises à jour des gradients calculés sur les paramètres du modèle.
                - global_parameters : Dictionnaire contenant les paramètres globaux du modèle. Ces paramètres sont chargés dans le modèle local avant la mise à jour. Cela permet au modèle local de commencer la mise à jour à partir d'une version récente du modèle global.
        """

        loss_fn = F.cross_entropy

        # Chargement des parametres dans le modele
        net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(
            self.train_ds, batch_size=local_batch_size, shuffle=True)  # Chargement du dataset d'entrainement

        for epoch in range(local_epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.device), label.to(self.device)
                preds = net(data)  # Effectue la prédiction
                # Calcule la perte en comparant la prédiction au labele
                loss = loss_fn(preds, label)
                loss.backward()  # Effectue la rétropropagation
                optimizer.step()  # Applique la mise à jour du poids
                optimizer.zero_grad()  # Reinitialise les gradients du modèle

        return net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, num_of_clients: int, is_iid: bool = True, device: str = "cpu"):
        self.is_iid = is_iid
        self.num_of_clients = num_of_clients
        self.device = device
        self.clients_set = {}

        self.test_data_loader = None

        self.balance_dataset()

    def balance_dataset(self):
        """
            Méthode pour répartir équitablement les données d'entraînement entre les clients. \n 
            Elle utilise la classe GetDataSet pour obtenir les données MNIST et les divise en lots équilibrés entre les clients.
        """
        mnist_dataset = GetDataSet(self.is_iid)

        test_data = torch.tensor(mnist_dataset.test_data)
        test_label = torch.argmax(torch.tensor(
            mnist_dataset.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(
            test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label

        shard_size = mnist_dataset.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(
            mnist_dataset.train_data_size // shard_size)

        for i in range(self.num_of_clients):
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

            self.clients_set['client{}'.format(i)] = someone
