import random
import numpy as np
from collections import Counter

class Game:
    def __init__(self):
        self.board = np.zeros((5, 5), dtype=int)

    def roll_dice(self):
        return random.randint(1, 6) + random.randint(1, 6)

    def get_empty_cells(self):
        return list(zip(*np.where(self.board == 0)))
    
    def end_game(self):
        return len(self.get_empty_cells()) == 0
    
    def reset_game(self):
        self.board.fill(0)

    def place_number(self, row, col, number):
        if row < 0 or row > 4 or col < 0 or col > 4: return False
        if self.board[row, col] == 0:
            self.board[row, col] = number
            return True
        return False

    def display_board(self):
        print(self.board)

    def get_score(self, line):
        if 0 in line: return 0 # not filled yet
        counts = Counter(line).values()
        sorted_counts = sorted(counts, reverse=True)
        unique_nums = sorted(set(line))
        
        # check for straight
        is_straight = len(unique_nums) == 5 and (max(unique_nums) - min(unique_nums) == 4)
        if is_straight:
            return 8 if 7 in line else 12
        
        # check for combinations
        if 5 in sorted_counts: return 10  # 5er
        if 4 in sorted_counts: return 6   # 4er
        if 3 in sorted_counts and 2 in sorted_counts: return 8 # Full House
        if 3 in sorted_counts: return 3   # 3er
        if sorted_counts.count(2) == 2: return 3 # 2er + 2er
        if 2 in sorted_counts: return 1   # 2er
        
        return 0

    def calculate_total_score(self):
        total = 0
        # rows & columns
        for i in range(5):
            total += self.get_score(self.board[i, :])
            total += self.get_score(self.board[:, i])
        # diagonals (double score)
        total += self.get_score(np.diagonal(self.board)) * 2
        total += self.get_score(np.diagonal(np.fliplr(self.board))) * 2
        return total

# dicegame = Game()

# while not dicegame.end_game():
#     dice = dicegame.roll_dice()
#     empty_cells = dicegame.get_empty_cells()
#     if empty_cells:
#         cell = random.choice(empty_cells)
#         dicegame.place_number(cell[0], cell[1], dice)
#         dicegame.display_board()
#         print(f"Score: {dicegame.calculate_total_score()}")