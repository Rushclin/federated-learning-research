# import os
# import subprocess

# # Assurez-vous que le chemin vers le script est correct et que le script est exécutable
# script_path = "./commands/ResNet/fedavg-iid.sh"

# # Lancer le script en arrière-plan
# # os.spawnl(os.P_DETACH, script_path)
# subprocess.Popen([script_path])



import subprocess

# Informations de connexion SSH
ssh_user = "ml_admin"
ssh_host = "8.tcp.eu.ngrok.io"
ssh_key_path = "C:/Users/Rushclin02/.ssh/id_rsa" 
remote_script_path = "./commands/ResNet/fedavg-iid.sh"

# Commande SSH pour lancer le script avec nohup
ssh_command = f'ssh -i {ssh_key_path} {ssh_user}@{ssh_host} "nohup {remote_script_path} > /dev/null 2>&1 &"'

# Lancer la commande SSH avec subprocess
process = subprocess.Popen(ssh_command, shell=True)
process.wait()

print("Script lancé en arrière-plan sur la machine distante")
