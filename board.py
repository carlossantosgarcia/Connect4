import numpy as np
import pygame
import sys
import math
import pandas as pd
import time
from alpha_beta_pruning import static, children, moves, minimax, best_move, moves, Q_children
from random import choice

# Constants
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)


class Connect4:
    ''' Some methods assume that the board has a 6x7 size'''

    def __init__(self, nb_rows=6, nb_cols=7):
        self.rows = nb_rows
        self.cols = nb_cols
        self.board = np.zeros((nb_rows, nb_cols))
        self.game_over = False
        self.turn = 0
        self.winner = None

    def play_move(self, row, col, piece):
        self.board[row][col] = piece

    def move_is_valid(self, col):
        return self.board[self.rows-1][col] == 0

    def get_available_row(self, col):
        for r in range(self.rows):
            if self.board[r][col] == 0:
                return r

    def print_board(self):
        '''Flips the board to match the real game'''
        print(np.flip(self.board, 0))

    def check_wins(self, piece):
        # Check horizontal locations for win
        for c in range(self.cols-3):
            for r in range(self.rows):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.cols):
            for r in range(self.rows-3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(self.cols-3):
            for r in range(self.rows-3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(self.cols-3):
            for r in range(3, self.rows):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True

    def draw_board(self, screen):
        height = (self.rows+1) * SQUARESIZE

        for c in range(self.cols):
            for r in range(self.rows):
                pygame.draw.rect(
                    screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, BLACK, (int(
                    c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)

        for c in range(self.cols):
            for r in range(self.rows):
                if self.board[r][c] == 1:
                    pygame.draw.circle(screen, RED, (int(
                        c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif self.board[r][c] == 2:
                    pygame.draw.circle(screen, YELLOW, (int(
                        c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()

    def play(self):

        pygame.init()

        width = self.cols * SQUARESIZE
        height = (self.rows+1) * SQUARESIZE
        size = (width, height)

        screen = pygame.display.set_mode(size)
        self.draw_board(screen)
        pygame.display.update()

        myfont = pygame.font.Font('sweet purple.ttf', 75)

        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if self.turn == 0:
                        col = best_move(self)
                        row = self.get_available_row(col)
                        self.play_move(row, col, 1)

                        if self.check_wins(1):
                            label = myfont.render("Player 1 wins!", 1, RED)
                            screen.blit(label, (width/4, 10))
                            self.game_over = True
                            self.draw_board(screen)
                            break
                        self.draw_board(screen)
                        self.turn = 1 - self.turn

                    else:
                        pygame.draw.circle(
                            screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
                    pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))

                    # Ask for Player's input
                    if self.turn == 0:
                        pass

                    else:
                        posx = event.pos[0]
                        col = int(posx/SQUARESIZE)
                        if self.move_is_valid(col):
                            row = self.get_available_row(col)
                            self.play_move(row, col, 2)

                            if self.check_wins(2):
                                label = myfont.render(
                                    "Player 2 wins!", 1, YELLOW)
                                screen.blit(label, (width/4, 10))
                                self.game_over = True
                                self.draw_board(screen)
                                break

                        self.draw_board(screen)
                        self.turn = 1 - self.turn

        pygame.time.wait(5000)

    def board_to_string(self):
        s = ''
        for row in range(self.rows):
            for col in range(self.cols):
                s += str(int(self.board[row][col]))
        return s

    def string_to_board(self, board_string):
        for row in range(self.rows):
            for col in range(self.cols):
                self.board[row][col] = float(board_string[self.cols*row+col])

    def train_q_learning(self, qdict):
        '''Modifies winner attribute'''
        ALPHA = 0.5
        GAMMA = 0.9
        while not self.game_over:
            if self.turn == 0:
                # Minimax plays here as Player 1
                col = best_move(self)
                row = self.get_available_row(col)
                self.play_move(row, col, 1)
                if self.check_wins(1):
                    self.game_over = True
                    self.winner = 0
                self.turn = 1 - self.turn
            else:
                # Chosen move has the highest q-value
                state = self.board_to_string()
                coups = moves(self)
                current_children, minimax_moves = Q_children(self)

                max_q_value = -np.inf
                chosen_column = 0
                reward = 0

                try:
                    qdict[state]
                except:
                    # State gets added to the table
                    qdict[state] = np.random.uniform(-0.01, 0.01)

                # Current state q-value
                Q = qdict[state]

                # Random number to allow some exploitation
                eps = np.random.uniform(0, 1)

                if eps < 0.1:
                    # Exploration
                    chosen_column = choice(list(coups))
                    # We here look for the maximum q-value among the current state's children
                    while max_q_value == -np.inf:
                        for playable_move in current_children.keys():
                            try:
                                if qdict[current_children[playable_move].board_to_string()] > max_q_value:
                                    max_q_value = qdict[current_children[playable_move].board_to_string(
                                    )]
                            except:
                                qdict[current_children[playable_move].board_to_string(
                                )] = np.random.uniform(-0.01, 0.01)
                else:
                    # Exploitation
                    # Here again we look for the maximum q-value among the current state's children
                    while max_q_value == -np.inf:
                        for playable_move in current_children.keys():
                            try:
                                if qdict[current_children[playable_move].board_to_string()] > max_q_value:
                                    max_q_value = qdict[current_children[playable_move].board_to_string(
                                    )]
                                    chosen_column = int(playable_move)
                            except:
                                qdict[current_children[playable_move].board_to_string(
                                )] = np.random.uniform(-0.01, 0.01)

                previous_q_value = (1-ALPHA)*Q + ALPHA*(GAMMA*max_q_value)

                # Player 2 plays
                row = self.get_available_row(chosen_column)
                self.play_move(row, chosen_column, 2)
                self.turn = 0

                if self.check_wins(2):
                    reward = 1
                    qdict[state] = (1-ALPHA) * Q + ALPHA
                    self.winner = 1
                    self.game_over = True
                else:
                    move = minimax_moves[str(chosen_column)]
                    row = self.get_available_row(move)
                    self.play_move(row, move, 1)
                    self.turn = 1
                    if self.check_wins(1):
                        self.game_over = True
                        self.winner = 0
                        reward = -1
                        qdict[state] = previous_q_value + ALPHA*reward
                try:
                    self.winner
                except:
                    qdict[state] = previous_q_value + ALPHA*reward
