import copy
import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice


def static(Connect4):
    '''Evalue statiquement un plateau de jeu'''
    S = 0
    signe = {1: 1, 2: -1}
    good_positions = [{'1110', '1101', '1011', '0111'},
                      {'2220', '2202', '2022', '0222'}]
    win_positions = ['1111', '2222']

    for piece in [1, 2]:

        # Checks rows
        for r in range(Connect4.rows):
            ROW = str(int(Connect4.board[r][0]))
            for c in range(1, Connect4.cols):
                ROW += str(int(Connect4.board[r][c]))
            for start_index in range(len(ROW) - 3):
                if ROW[start_index:start_index + 4] in good_positions[piece-1]:
                    S += signe[piece]*10
                if ROW[start_index:start_index + 4] in win_positions[piece-1]:
                    S += signe[piece]*1000
        # Checks columns
        for c in range(Connect4.cols):
            COL = str(int(Connect4.board[0][c]))
            for r in range(1, Connect4.rows):
                COL += str(int(Connect4.board[r][c]))
            for start_index in range(len(COL) - 3):
                if COL[start_index:start_index + 4] in good_positions[piece-1]:
                    S += signe[piece]*10
                if COL[start_index:start_index + 4] in win_positions[piece-1]:
                    S += signe[piece]*1000

        # Checks positively sloped diagonals
        for c in range(Connect4.cols-3):
            for r in range(Connect4.rows-3):
                DIAG = str(int(Connect4.board[r][c]))
                for i in range(1, 4):
                    DIAG += str(int(Connect4.board[r+i][c+i]))
                for start_index in range(len(DIAG) - 3):
                    if DIAG[start_index:start_index + 4] in good_positions[piece-1]:
                        S += signe[piece]*10
                    if DIAG[start_index:start_index + 4] in win_positions[piece-1]:
                        S += signe[piece]*1000

        # Checks negatively sloped diagonals
        for c in range(Connect4.cols-3):
            for r in range(3, Connect4.rows):
                DIAG = str(int(Connect4.board[r][c]))
                for i in range(1, 4):
                    DIAG += str(int(Connect4.board[r-i][c+i]))
                for start_index in range(len(DIAG) - 3):
                    if DIAG[start_index:start_index + 4] in good_positions[piece-1]:
                        S += signe[piece]*10
                    if DIAG[start_index:start_index + 4] in win_positions[piece-1]:
                        S += signe[piece]*1000

    return S


def children(Connect4):
    '''Returns a dictionnary {move:obtained_child}'''
    dict_children = {}
    for col in range(Connect4.cols):
        if Connect4.move_is_valid(col):
            Child = copy.deepcopy(Connect4)
            row = Child.get_available_row(col)
            children_piece = Connect4.turn + 1
            Child.play_move(row, col, children_piece)
            Child.turn = 1 - Connect4.turn
            if Child.check_wins(children_piece) or not moves(Child):
                Child.game_over = True
            dict_children[str(col)] = Child
    if not dict_children and not Connect4.game_over:
        Connect4.game_over = True
    return dict_children


def Q_children(Connect4):
    '''Returns a dictionnary {move:obtained_child}

    Takes into account Minimax's move in between.
    '''
    dict_children = {}
    minimax_moves = {}
    for col in range(Connect4.cols):
        if Connect4.move_is_valid(col):
            Child = copy.deepcopy(Connect4)
            row = Child.get_available_row(col)
            children_piece = Connect4.turn + 1
            Child.play_move(row, col, children_piece)
            Child.turn = 1 - Connect4.turn
            if Child.check_wins(children_piece):
                Child.game_over = True
            else:
                col2 = best_move(Child)
                minimax_moves[str(col)] = col2
                row2 = Child.get_available_row(col2)
                piece = Child.turn + 1
                Child.play_move(row2, col2, piece)
                Child.turn = 1 - Connect4.turn
                if Child.check_wins(piece):
                    Child.game_over = True

            dict_children[str(col)] = Child
    if not dict_children and not Connect4.game_over:
        Connect4.game_over = True
    return dict_children, minimax_moves


def moves(Connect4):
    '''Returns an array of available moves from current state'''
    L = set()
    for col in range(Connect4.cols):
        if Connect4.move_is_valid(col):
            L.add(col)
    if not L:
        Connect4.game_over = True
    return L


def minimax(Connect4, depth, alpha, beta, maximizingPlayer=None):
    '''Returns best possible score obtained from the root'''
    if maximizingPlayer == None:
        if not Connect4.turn:
            maximizingPlayer = True
        else:
            maximizingPlayer = False

    if not depth or Connect4.game_over:
        return static(Connect4)

    elif maximizingPlayer:
        maxEval = -float('inf')
        for Child in children(Connect4).values():
            eval = minimax(Child, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = float('inf')
        for Child in children(Connect4).values():
            eval = minimax(Child, depth - 1, alpha, beta, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def best_move(Connect4):
    '''Returns best possible move from current state, based on minimax algorithm'''
    # Player 1 has here turn 0 and plays as MAX
    if not Connect4.game_over:
        coups = moves(Connect4)
        if not Connect4.turn:
            # Turn = 0
            scores = [-float('inf')]*Connect4.cols
        else:
            # Turn = 1
            scores = [float('inf')]*Connect4.cols

        fils = children(Connect4)
        for playable_move in fils.keys():
            scores[int(playable_move)] = minimax(fils[playable_move], 3, -float('inf'),
                                                 float('inf'), not fils[playable_move].turn)

        hyp_move = scores.index(max(scores))

        # Not playable moves get NaN as a score
        for col in range(Connect4.cols):
            if col not in coups:
                scores[col] = float('nan')

        if Connect4.turn == 0:
            # Children score is then maximized
            if abs(np.nanmean(scores)) != float('inf'):
                if max(scores) == int(np.nanmean(scores)):
                    if not max(scores) and 3 in coups:
                        move = 3
                        return move
                    else:
                        move = choice(list(coups))
                        return move
                else:
                    move = hyp_move
                    return move
            else:
                Connect4.print_board()
                print('Coups', coups)
                print('Scores', scores)
                print('Game over', Connect4.game_over)
                bug_file = open('bug_reports.txt', 'a')
                bug_file.write('Coups: '+str(list(coups))+'\n')
                bug_file.write('Scores: '+str(scores)+'\n')
                bug_file.write('Board: '+Connect4.board_to_string()+'\n')
                bug_file.write('Game Over: ' + str(Connect4.game_over) + '\n')
                bug_file.write(' ')
                bug_file.close()
                move = choice(list(coups))
                return move
        else:
            # Children score is minimized
            move = scores.index(min(scores))
            return move
