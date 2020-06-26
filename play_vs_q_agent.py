from board import *
import ast

game = Connect4()

# Upload values from table
Q_table = {}
table = pd.read_csv('q_learning_table.csv')
for i in range(len(table['states'])):
    qdict[table['states'][i]] = ast.literal_eval(table['scores'][i])

game.vs_q_play(Q_table)
