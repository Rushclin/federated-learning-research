import numpy as np
import torch
import hydra
import warnings

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch import optim

from model import Model_2NN, Model_CNN
from clients import ClientsGroup

DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

# Ignorer tous les avertissements (notamment les UserWarnings)
warnings.filterwarnings("ignore")


@hydra.main(config_name="base", config_path="conf", version_base=None)
def main(cfg: DictConfig):

    config = OmegaConf.to_object(cfg)

    # Recuperation de la configuration.
    save_freq = config['save_freq']  # Frequence de sauvegarde
    num_of_clients = int(config['num_of_clients'])  # Nombre de client
    # Learning rate, taux d'apprentissage
    learning_rate = config['learning_rate']
    iid = config['iid']  # La distribution sera iid ?
    epoch = config['epoch']  # Le nombre d'epoque
    num_comm = config['num_comm']  # Le nombre de tour de formation
    val_freq = config['val_freq']  # Frequence de validation
    batchsize = config['batchsize']
    model = config['model']
    cfraction = config['cfraction']

    net = None

    if model == "Model_2NN":
        net = Model_2NN()
    else: 
        net = Model_CNN()

    net = net.to(DEVICE) # On indique sur quel device on doit executer l'algorithme. CPU ou GPU

    opti = optim.SGD(net.parameters(), lr=learning_rate)

    my_clients = ClientsGroup(num_of_clients, DEVICE)
    test_data_loader = my_clients.test_data_loader

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(num_comm):
        print("Tour de communication {}".format(i+1))

        num_in_comm = int(max(num_of_clients * cfraction, 1))
        order = np.random.permutation(num_of_clients)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = my_clients.clients_set[client].local_update(
                local_epoch=epoch,
                local_batch_size=batchsize,
                net=net,
                optimizer=opti,
                global_parameters=global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + \
                        local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % val_freq == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in test_data_loader:
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean() # chercher a comprendre pourquoi on a utilise la moyenne. 
                    num += 1
                print('Taux d\'apprentissage: {}'.format(sum_accu / num))


if __name__ == "__main__":
    main()  # Démarrage de la fonction
