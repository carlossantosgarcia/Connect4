from board import *

game = Connect4()

# Update values from table
qdict = {}
table = pd.read_csv('q_table.csv')
for i in range(len(table['states'])):
    qdict[table['states'][i]] = table['scores'][i]

game.vs_q_play(qdict)
