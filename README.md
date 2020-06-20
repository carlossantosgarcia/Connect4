# Installation
Les modules nécessaires pour éxécuter les différents scripts concernant Puissance 4 sont:
* Pygame
* Pandas
* Numpy

# Utilisation:
Pour jouer une partie de puissance 4 contre l'IA utilisant l'élagage Alpha-Bêta:
```python
python play_connect_4.py
```

# Organisation du projet
## Classe Connect4()
Il s'agit de la classe principale du projet, ses attributs principaux sont le plateau de jeu (_board_), ses dimensions et lequel des joueurs doit jouer (_turn_). Les méthodes de la classe permettent de joueur au jeu.

## Algorithme d'élagage Alpha-Bêta
Permet une recherche arborescente à profondeur fixée pour trouver le meilleur coup de l'agent. La fonction _best_move_ l'implémente dans le cas de Puissance 4.

## Q-learning
On a ici considéré un agent 1 qui commence à jouer et joue en suivant l'algorithme d'élagage Alpha-Bêta contre un agent 2 qui actualise sa table de données stockée sous forme de fichier csv. Pour cela, il s'appuie sur la fonction _Q_children_ qui prend en compte le coup de l'agent 1 entre chaque coup de l'agent 2 pour passer d'un état où l'agent doit jouer au suivant. L'idée était ici que l'agent 2 apprenne à jouer contre l'agent 1 et finisse par battre Minimax.
La fonction _train_q_learning_ permet aux deux agents de jouer. Le script **_q_learning.py_** permet de mettre à jour le dernier fichier csv qui enregistre les différents par lesquels le jeu est passé: le fichier est composé d'une colonne de 'states', ie. les différents plateaux du jeu transformés en string, ainsi qu'une colonne avec leurs différents 'scores' qui sont mis à jour. Par exemple:

| States | Scores |
| ----- | -----: |
|201210020221000011000001000000000000000000|0.3|
|201210000221000011200001000000000000000000|0.1|
|201110200000000000000000000000000000000000|-0.7|

Le fichier`q_learning_server.py` a été conçu pour être executé dans le cloud, et enregistrer les données toutes les X parties. Le fichier `games.csv` permet de suivre le nombre de parties effectuées. Les dernières données accessibles des parties sont dans le fichier `q_table.csv`.
