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
Pour jouer une partie de puissance 4 contre l'agent entraîné au Q-learning:
```python
python play_vs_q_agent.py
```

# Organisation du projet
## Classe Connect4
Il s'agit de la classe principale du projet, ses attributs principaux sont le plateau de jeu (_board_), ses dimensions et lequel des joueurs doit jouer (_turn_). Les méthodes de la classe permettent le déroulement d'une partie.

## Algorithme d'élagage Alpha-Bêta
Permet une recherche arborescente à profondeur fixée pour trouver le meilleur coup de l'agent. La fonction _best_move_ l'implémente dans le cas de Puissance 4.

## Q-learning
On a ici considéré un agent 1 qui commence à jouer et joue en suivant l'algorithme d'élagage Alpha-Bêta contre un agent 2 qui actualise sa table de données stockée sous forme de fichier csv. Pour cela, il s'appuie sur la fonction `Q_children` qui prend en compte le coup de l'agent 1 entre chaque coup de l'agent 2 pour passer d'un état où l'agent doit jouer au suivant. L'idée était ici que l'agent 2 apprenne à jouer contre l'agent 1 et finisse par battre Minimax.
La fonction `train_q_learning` permet aux deux agents de jouer. Le script `q_learning.py` permet de mettre à jour le dernier fichier csv qui enregistre les différents par lesquels le jeu est passé: le fichier est composé d'une colonne de 'states', ie. les différents plateaux du jeu transformés en string, ainsi qu'une colonne avec leurs différents listes de 'scores' qui sont mis à jour, et qui représentent, pour l'indice `i`, la valeur de jouer l'action `i` à l'état donné. Voici à quoi ressemble cette table:

| States | Scores |
| ----- | -----: |
|201210020221000011000001000000000000000000|[0, 0, 0.03656, 0, 0, 0, 0]|
|201210000221000011200001000000000000000000|[0, 0, -1, 0, -1, 0, 0]|
|201110200000000000000000000000000000000000|[0, -1, 0, 0.0465, 0, 0, 0]|

Le fichier`cloud_training.py` a été conçu pour entraîner notre agent, et enregistrer les données toutes les `GAMES` parties. Le fichier `games.csv` permet de suivre le nombre de parties effectuées, pourcentages de victoires, durées d'entraînement.... Les dernières données accessibles des parties sont dans le fichier `q_learning_table.csv`.
