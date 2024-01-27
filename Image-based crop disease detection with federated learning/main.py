import torch
import warnings
import hydra
import numpy as np

from tqdm import tqdm
from torch import optim
from omegaconf import DictConfig, OmegaConf

from federated.client import Client, ClientGroup

from utils import get_model 

warnings.filterwarnings("ignore")

from federated.dataset import get_dataset

DEVICE = "gpu" if torch.cuda.is_available() else "cpu" 

@hydra.main(config_name="base", config_path="conf", version_base=None)
def main(cfg: DictConfig): 

    print("Implementation de l'article Image-based crop disease detection with federated learning")

    config = OmegaConf.to_object(cfg)

    client = config['client']
    epoch = config['epoch']
    batch_size = config['batch_size']
    com_round = config['com_round']
    learning_rate = config['learning_rate']
    is_iid = config['is_iid']
    model = config['model']
    cfraction = config['cfraction'] 
    val_freq = config['val_freq'] 

    model = get_model(model, num_classes=36) # TODO: Remove 36

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    clients = ClientGroup(
        num_clients=client, 
        batch_size=batch_size,
        non_iid=is_iid,
        data_dir=r"./data"
    )

    train_loader, val_loader, classes = get_dataset(batch_size=batch_size, root=r"./data")

    global_parameters = {}
    for key, var in model.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(com_round):
        print("Tour de communication {}".format(i+1))

        num_in_comm = int(max(client * cfraction, 1))
        order = np.random.permutation(client)
        clients_in_comm_1 = ['client{}'.format(i) for i in order[0:num_in_comm]]
        clients_in_comm = order[0:num_in_comm]
        

        sum_parameters = None
        for client in tqdm(range(num_in_comm)): 
            local_parameters = clients.list_clients[client].update_local_model(
                model=model, 
                optimizer = optimizer,
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
                for data, label in val_loader:
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    preds = model(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('Taux d\'apprentissage: {}'.format(sum_accu / num))

if __name__ == "__main__": 
    main()