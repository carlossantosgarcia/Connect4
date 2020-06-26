from board import *
from subprocess import PIPE, Popen
from datetime import datetime
import ast

GAMES = 100


def write_games(n_games, wins):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M").replace(' ', ',')
    process = Popen(['tail', '-n', '1', 'games.csv'], stdout=PIPE)
    last_games = int(process.stdout.readlines()[
                     0].decode('UTF-8').split(',')[-1])

    with open('games.csv', 'a') as games_file:
        games_file.write(f'\n{dt_string},{wins},{last_games + n_games}')


qdict = {}
table = pd.read_csv('q_learning_table.csv')

for i in range(len(table['states'])):
    qdict[table['states'][i]] = ast.literal_eval(table['scores'][i])


while True:
    wins = lost = 0
    times = []
    max_time = 0
    min_time = float('inf')

    for i in range(1, GAMES+1):
        print('Game', f'{i}:')
        start = time.time()
        game = Connect4()
        game.train_q_learning(qdict)
        end = time.time()
        time_spent = float(f"{end-start:.1f}")
        print('Delta_t:', f'{time_spent}s')
        if game.winner == 1:
            wins += 1
        else:
            lost += 1

        times.append(time_spent)

        if time_spent > max_time:
            max_time = time_spent

        if time_spent < min_time:
            min_time = time_spent

    print('Victoires', wins)
    print('AVG:', f"{sum(times)/len(times):.1f}",
          'MAX:', max_time, 'MIN:', min_time)

    d = {'states': [], 'scores': []}
    df = pd.DataFrame(d)
    states, scores = [], []
    for i in qdict.keys():
        states.append(i)
        scores.append(qdict[i])
    df["states"] = states
    df["scores"] = scores
    df.to_csv('q_learning_table.csv', index=False)
    write_games(GAMES, wins)
