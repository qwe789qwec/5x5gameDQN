import random
import numpy as np
from collections import Counter

class gameEnv:
    def __init__(self):
        self.board = np.zeros((5, 5), dtype=int)
        self.current_dice = self.roll_dice()

    def roll_dice(self):
        return random.randint(1, 6) + random.randint(1, 6)

    def get_valid_actions(self):
        return [(r, c) for r in range(5) for c in range(5) if self.board[r, c] == 0]
    
    def end_game(self):
        return len(self.get_valid_actions()) == 0
    
    def reset(self):
        self.board.fill(0)
        self.current_dice = self.roll_dice()
        return self.board.copy(), self.current_dice

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
    
    def step(self, action):
        row, col = action // 5, action % 5
        # old_score = self.calculate_total_score()
        valid_move = self.place_number(row, col, self.current_dice)
        
        # 3. 處理無效動作 (給予極大懲罰並提早結束，避免無窮迴圈)
        if not valid_move:
            return (self.board.copy(), self.current_dice), -100, True 
        
        new_score = self.calculate_total_score()

        # 4. 判斷遊戲是否結束與計分
        done = self.end_game()
        # 採用稀疏獎勵：只有遊戲結束才給總分，過程給 0 分
        reward = self.calculate_total_score() if done else 0 
        # reward = new_score - old_score
        
        # 5. 如果遊戲還沒結束，擲下一次骰子準備給下一個 State
        if not done:
            self.current_dice = self.roll_dice()
        else:
            self.current_dice = 0
            # print(f"遊戲結束！最終分數: {new_score}")
            
        return (self.board.copy(), self.current_dice), reward, done