import hydra
import torch
import torch.nn.functional as F 

from torch import optim

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from models.model import Model

DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

"""

    Pour cette tache, on utilisera un perceptron multicouche (donnee MNIST) avec 2 couches cachees de 200 unite chacunes et une focntion d'activation ReLu

    Un CNN avec 2 couches de convolution de 5*5 (32, 64 cannaux) suivie d'un poolmax de 2*2, une couche entierrement connecte de 512 unite d'activation ReLu, une couche de sortie softmax

    Nombre de client : 100
    Taille de donnees par client: 600 dont 500 exemples de formation et 100 exemples de test

    Ils ont atteint 96.5% du taux de precision, mais l'objectif n'est pas de l'atteindre, mais d'evaluer l'optimisation FedAVG 


"""


# We hydrate with config file named conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig): 
    
    print("We are launch our application based on Federated Learning (FL)")
    print("\n Our configurations")

    cfg = OmegaConf.to_object(config) # Nous récupérons la configuration qui se trouve dans le fichier conf/base.yaml
    # Je dois ici parser toute la configuration 

    learning_rate = cfg['lr']


    net = Model() # Récupération du modèle
    net.to(DEVICE) # On indique sur quel device la formation ou les opérations seront appliquées

    loss_fn = F.cross_entropy # On utilisera la fonction entropy pour evaluer la perte 
    optimizer = optim.SGD(net.parameters(), lr=learning_rate) # On utilise une descente de gradient stochastique dans ce cas

    


if __name__ == "__main__": 
    # Launch main function 
    main()