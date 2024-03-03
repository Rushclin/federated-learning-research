import os
import sys
import time
import torch
import hydra
import traceback
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from importlib import import_module

from src import set_seed, set_logger, check_args, load_model, load_dataset, tensorboard_runner


# Pour le lancement de tensorboard
def board(args: DictConfig):
    tensorboard_runner(args)


@hydra.main(config_name="base", config_path="./src/config", version_base=None)
def main(args: DictConfig):
    """Programme principal pour lancer le federated learning.

    Args:
        args: Les argumenets qui se trouvent dans le fichier de configuration
    """

    # Initialisation de la sauvegarde
    # ------------ Debut ------------
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(
        args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log')

    writer = SummaryWriter(log_dir=os.path.join(
        args.log_path, f'{args.exp_name}_{curr_time}'), filename_suffix=f'_{curr_time}')

    # ------------ Fin ------------

    # ------ Board ----------
    board(args)

    # Modification du seed global de l'algorithme
    set_seed(args.seed)

    # Chargement du Dataset

    server_dataset, client_datasets = load_dataset(args)

    # Verification de tous les arguments ===> Ici on verifie s'ils sont vrais
    args = check_args(args)

    # On doit charger le modele ici
    model = load_model(args)

    # On doit charger le server ici
    server_class = import_module(f"src.servers.{args.algorithm}server").__dict__[
        f'{args.algorithm.title()}Server']
    
    # Initialisation du serveur
    server = server_class(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets, model=model)

    # Federated learning
    for curr_round in range(1, args.R + 1):
        server.round = curr_round

        selected_ids = server.update() 

        if (curr_round % args.eval_every == 0) or (curr_round == args.R):
            server.evaluate(excluded_ids=selected_ids)
    else:
        server.finalize()

if __name__ == "__main__":

    # Lancer le programme principal
    torch.autograd.set_detect_anomaly(True)
    try:
        main()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
