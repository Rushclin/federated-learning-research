# Federated Learning Research
I create this repository to learn more about FL (Federated Learning). In this course, I will learn Federated Learning with Flower, from the basics to understanding the field well.


## Définition
Le Federated Learning (FL) est un cadre d'apprentissage décentralisé qui permet de tirer partie de nombreuse bases de données pour entrainer un modèle d'IA sans avoir à collecter les données sur le même appareil (ceci peut être un serveur).

Plutot que d'envoyer les données à l'algorithme, on fait venir l'algorithme vers les données et ne faire remonter que les gradients (paramètre dans d'autre cas) au serveur pour une agrégation. 


## Problematique
Avec la prise de conscience croissante sur la confidentialité, les dévéloppeurs de solutions basé sur IA ne doivent pas ignorer le fait que leur modèle accède aux données sensibles de l’utilisateur. 

C’est donc là qu’intevient le Federated Learning 


## Etape du Federated Learning (FL)

1. Nous créons notre modèle qui est hébergé sur un serveur centralisé, chaque appareil télécharge le modèle (une copie locale) où ce dernier pourra s’exécuter.
2. Le modèle apprend et s’entraine sur les données locale de l’utilisateur,
3. Une fois entraînés, les appareils sont autorisés à synchroniser avec le serveur central afin de transférer le résultat local vers le serveur central.
4. Notons que seuls les résultats sont transférés et non les données utilisateur. 

## NB
Dans notre cas, nous allons essayer de reproduire les articles lu jusqu'a present et les reproduire afin d'arriver a mieux comprendre le domaine. 
