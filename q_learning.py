from board import *

qdict = {}
table = pd.read_csv('q_table.csv')
for i in range(len(table['states'])):
    qdict[table['states'][i]] = table['scores'][i]

win, lose = 0, 0
start = time.time()
for i in range(700):
    print(i)
    game = Connect4()
    game.train_q_learning(qdict)
    if game.winner == 1:
        win += 1
    elif game.winner == 0:
        lose += 1
end = time.time()
print('Delta_t: ', end-start)
print('Victoires ', win)

d = {'states': [], 'scores': []}
df = pd.DataFrame(d)
states, scores = [], []
for i in qdict.keys():
    states.append(i)
    scores.append(qdict[i])
df["states"] = states
df["scores"] = scores
df.to_csv('q_table.csv')
