import hydra
import torch
import torch.nn.functional as F 
import numpy as np

from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from torch import optim

from model import Model
from clients import ClientsGroup

DEVICE = 'cpu' if not torch.cuda.is_available() else 'gpu'

@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig): 

    print("\n Périphérique :", DEVICE, " Version de PyTorch", torch.__version__ ,"\n")

    config = OmegaConf.to_object(cfg)

    # Recuperation de la configuration.
    save_freq = config['save_freq'] # Frequence de sauvegarde 
    num_of_clients = config['num_of_clients'] # Nombre de client
    learning_rate = config['learning_rate'] # Learning rate, tauc d'apprentissage
    iid = config['iid'] # La distribution sera iid ?
    epoch = config['epoch'] # Le nombre d'epoque
    num_comm = config['num_comm'] # Le nombre de tour de formation
    val_freq = config['val_freq'] # Frequence de validation
    batchsize = config['batchsize']

    net = Model()
    net.to(DEVICE) # On lui donne sur quel device il doit devoir s'executer. 

    loss_fn = F.cross_entropy
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    my_clients = ClientsGroup(is_iid=iid, num_of_clients=num_of_clients, device=DEVICE)
    test_data_loader = my_clients.test_data_loader

    num_in_comm = int(max((num_of_clients * 1), 1))

    global_parameters = {}
    for key, var in net.state_dict().items(): 
        global_parameters[key] = var.clone()
    
    for i in range(num_comm): 
        print(f"Tour de communication {i+1}")

        order = np.random.permutation(num_of_clients) 
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        
        for client in tqdm(clients_in_comm):
            local_parameters = my_clients.clients_set[client].local_update(
                net,
                epoch,
                batchsize,
                loss_fn,
                optimizer,
                global_parameters
            )

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        
        with torch.no_grad():
            if (i + 1) % 1 == val_freq: 
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in test_data_loader:
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('Taux d\'apprentissage : {}'.format(sum_accu / num))
            
        # Je peux maintenant ici sauvegarder le modele

if __name__ == '__main__': 
    main()