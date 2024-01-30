import torch
import hydra
import warnings
import os

import numpy as np

from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from torch import optim
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from clients import ClientGroup
from utils import get_model
from dataset import GetDataSet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")


@hydra.main(config_name="base", config_path="conf", version_base=None)
def main(cfg: DictConfig):

    # Repertoire de sauvegarde
    path = r"./weigths/"

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

    dataset = GetDataSet()

    model = get_model(model, num_classes=dataset.classes_size, device=DEVICE)

    load_model = model.model

    optimizer = optim.SGD(load_model.fc.parameters(), lr=learning_rate)

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

        # Evaluation du modèle
        with torch.no_grad():
            if (i + 1) % val_freq == 0:
                model.load_state_dict(global_parameters, strict=True)

                # Variables pour les métriques
                sum_accu = 0
                sum_precision = 0
                sum_recall = 0
                sum_f1 = 0
                num = 0

                for data, label in test_data_loader:
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    preds = model(data)
                    preds = torch.argmax(preds, dim=1)

                    # Accuracy
                    accu = accuracy_score(label, preds)
                    sum_accu += accu

                    # Calcul de la précision, du rappel et du F1-score
                    precision = precision_score(label, preds, average='macro')
                    recall = recall_score(label, preds, average='macro')
                    f1 = f1_score(label, preds, average='macro')

                    sum_precision += precision
                    sum_recall += recall
                    sum_f1 += f1

                    num += 1

                avg_accuracy = sum_accu / num
                avg_precision = sum_precision / num
                avg_recall = sum_recall / num
                avg_f1 = sum_f1 / num

                print('Accuracy - Taux d\'apprentissage : {}'.format(avg_accuracy))
                print('Precision - Précision : {}'.format(avg_precision))
                print('Recall - Rappel : {}'.format(avg_recall))
                print('F1-Score - Score F1 : {}'.format(avg_f1))

                # Ecriture dans un fichier txt de toutes les metriques

                with open(os.path.join(path, "evaluation_info.txt"), "a") as f:
                    f.write(f"Communication Round: {i+1}\n")
                    f.write(f"Number of Epochs: {epoch}\n")
                    f.write(f"Accuracy: {avg_accuracy.item()}\n")
                    f.write(f"Precision: {avg_precision}\n")
                    f.write(f"Recall: {avg_recall}\n")
                    f.write(f"F1-Score: {avg_f1}\n")
                    f.write(f"client: {client}\n")
                    f.write(f"learning_rate: {learning_rate}\n")
                    f.write(f"cfraction: {cfraction}\n")

        

    # Sauvegarder le modèle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path, "modele.h5"))
    print("Le modèle a été sauvegardé avec succès dans le répertoire : {}".format(save_path))

if __name__ == "__main__":
    main()  # Demarrer l'application
