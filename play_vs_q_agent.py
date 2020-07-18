from board import *
from pandas import read_csv
import ast


game = Connect4()

# Upload values from table
Q_table = {}
table = read_csv('q_learning_table.csv')
for i in range(len(table['states'])):
    Q_table[table['states'][i]] = ast.literal_eval(table['scores'][i])

game.vs_q_play(Q_table)
