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
    '''Renvoie un dictionnaire {coups:fils_obtenu}'''
    a = {}
    for i in range(Connect4.cols):
        if Connect4.move_is_valid(i):
            Child = copy.deepcopy(Connect4)
            r = Child.get_available_row(i)
            children_piece = Connect4.turn + 1
            Child.play_move(r, i, children_piece)
            Child.turn = 1 - Connect4.turn
            if Child.check_wins(children_piece):
                Child.game_over = True
            a[str(i)] = Child
    if a == {} and not Connect4.game_over:
        Connect4.game_over = True
    return a


def Q_children(Connect4):
    '''Renvoie un dictionnaire {coups:fils_obtenu}

    Prends en compte le coup de minimax entre deux coups de notre agent
    '''
    a = {}
    for i in range(Connect4.cols):
        if Connect4.move_is_valid(i):
            Child = copy.deepcopy(Connect4)
            r = Child.get_available_row(i)
            children_piece = Connect4.turn + 1
            Child.play_move(r, i, children_piece)
            Child.turn = 1 - Connect4.turn
            if Child.check_wins(children_piece):
                Child.game_over = True
            else:
                col = best_move(Child)
                # Child.print_Connect4()
                row = Child.get_available_row(col)
                piece = Child.turn + 1
                Child.play_move(row, col, piece)
                Child.turn = 1 - Connect4.turn
                if Child.check_wins(piece):
                    Child.game_over = True

            a[str(i)] = Child
    if a == {} and not Connect4.game_over:
        Connect4.game_over = True
    return a


def moves(Connect4):
    '''Retourne une liste des coups possibles pour le jeu donné'''
    L = set()
    for i in range(Connect4.cols):
        if Connect4.move_is_valid(i):
            L.add(i)
    if not L:
        Connect4.game_over = True
    return L


def minimax(Connect4, depth, alpha, beta, maximizingPlayer=None):
    '''Retourne le meilleur score possible atteignable depuis la racine pour le joueur donné'''
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
    '''Renvoie le meilleur coup à jouer depuis une position en suivant l'algorithme minimax'''
    # Par convention, on maximise le score pour le joueur 1, ie turn = 0
    if not Connect4.game_over:
        coups = moves(Connect4)
        if not Connect4.turn:
            # Turn = 0
            scores = [-float('inf')]*Connect4.cols
        else:
            #Turn = 1
            scores = [float('inf')]*Connect4.cols
        fils = children(Connect4)
        for i in fils.keys():
            scores[int(i)] = minimax(fils[i], 3, -
                                     float('inf'), float('inf'), fils[i].turn == 0)

        hyp_move = scores.index(max(scores))

        for i in range(Connect4.cols):
            if i not in coups:
                scores[i] = float('nan')

        if Connect4.turn == 0:
            # On maximise alors le score des children
            if max(scores) == int(np.nanmean(scores)):
                if not max(scores) and 3 in coups:
                    move = 3
                else:
                    move = choice(list(coups))
            else:
                move = hyp_move
            return move
        else:
            # On minimise le score des children
            move = scores.index(min(scores))
            return move
    return None
