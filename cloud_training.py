from ast import literal_eval
from datetime import datetime
from subprocess import PIPE, Popen
from time import time
from pandas import DataFrame, read_csv
from board import Connect4
from numpy import PINF


GAMES = 10


def write_games(n_games, wins):
    dt_string = datetime.now().strftime('%d/%m/%Y %H:%M').replace(' ', ',')
    process = Popen(['tail', '-n', '1', 'games.csv'], stdout=PIPE)
    last_games = int(process.stdout.readlines()[
                     0].decode('UTF-8').split(',')[-1])

    with open('games.csv', 'a') as games_file:
        games_file.write(f'\n{dt_string},{wins},{last_games + n_games}')


qdict = {}
table = read_csv('q_learning_table.csv')

for i in range(len(table['states'])):
    qdict[table['states'][i]] = literal_eval(table['scores'][i])


while True:
    wins = lost = 0
    max_time = 0
    min_time = PINF
    average = 0

    for i in range(1, GAMES + 1):
        print(f'Game {i}:')
        start = time()
        game = Connect4()
        game.train_q_learning(qdict)
        end = time()
        time_spent = end - start
        print('Delta_t:', f'{round(time_spent, 1)}s')

        if game.winner == 1:
            wins += 1
        else:
            lost += 1

        average += time_spent / GAMES

        if time_spent > max_time:
            max_time = time_spent

        if time_spent < min_time:
            min_time = time_spent

    print('Victoires', wins)
    print(f'AVG: {round(average)}',
          f'MAX: {round(max_time)}', f'MIN: {round(min_time)}')

    d = {'states': [], 'scores': []}
    df = DataFrame(d)
    states, scores = [], []

    for i in qdict.keys():
        states.append(i)
        scores.append(qdict[i])

    df['states'] = states
    df['scores'] = scores
    df.to_csv('q_learning_table.csv', index=False)
    write_games(GAMES, wins)
