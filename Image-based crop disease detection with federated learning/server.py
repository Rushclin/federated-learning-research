import torch
import hydra
import warnings

import numpy as np

from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from torch import optim

from clients import ClientGroup
from utils import get_model
from dataset import GetDataSet

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore")


@hydra.main(config_name="base", config_path="conf", version_base=None)
def main(cfg: DictConfig):

    config = OmegaConf.to_object(cfg)

    num_of_clients = config['client']
    epoch = config['epoch']
    batch_size = config['batch_size']
    com_round = config['com_round']
    learning_rate = config['learning_rate']
    is_iid = config['is_iid']
    model = config['model']
    cfraction = config['cfraction']
    val_freq = config['val_freq']

    # Je vais ici recuperer la taille de mes classes, pour eviter de le faire a la main

    dataset = GetDataSet()

    model = get_model(model, num_classes=dataset.classes_size)
    model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    clients = ClientGroup(
        batch_size=batch_size,
        device=DEVICE,
        non_iid=is_iid,
        num_of_clients=num_of_clients
    )

    test_data_loader = clients.test_data_loader

    global_parameters = {}
    for key, var in model.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(com_round):
        print("Tour de communication {}".format(i+1))

        num_in_comm = int(max(int(num_of_clients) * cfraction, 1))
        order = np.random.permutation(num_of_clients)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = clients.clients_set[client].clientUpdate(
                model=model,
                optimizer=optimizer,
                global_parameters=global_parameters,
                num_epochs=epoch,
                batch_size=batch_size,
                device=DEVICE
            )

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
                model.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in test_data_loader:
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    preds = model(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('Taux d\'apprentissage: {}'.format(sum_accu / num))


if __name__ == "__main__":
    main()
