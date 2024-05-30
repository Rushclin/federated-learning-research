import os
import subprocess

# Assurez-vous que le chemin vers le script est correct et que le script est exécutable
script_path = "./commands/ResNet/fedavg-iid.sh"

# Lancer le script en arrière-plan
# os.spawnl(os.P_DETACH, script_path)
subprocess.Popen([script_path])
