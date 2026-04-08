import random
import numpy as np
from collections import Counter

# --- 遊戲規則設定 ---
PROBS = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1}
DICE_SIDES = sum(PROBS.values()) # 36
NUMS = list(PROBS.keys())
WEIGHTS = [PROBS[n]/DICE_SIDES for n in NUMS]

def get_score(line):
    """計算單行/列/對角線的分數"""
    if 0 in line: return 0 # 未填滿不計分
    counts = Counter(line).values()
    sorted_counts = sorted(counts, reverse=True)
    unique_nums = sorted(set(line))
    
    # 檢查順子 (Straight)
    is_straight = len(unique_nums) == 5 and (max(unique_nums) - min(unique_nums) == 4)
    if is_straight:
        return 8 if 7 in line else 12
    
    # 檢查相同數字組合
    if 5 in sorted_counts: return 10  # 5er
    if 4 in sorted_counts: return 6   # 4er
    if 3 in sorted_counts and 2 in sorted_counts: return 8 # Full House
    if 3 in sorted_counts: return 3   # 3er
    if sorted_counts.count(2) == 2: return 3 # 2er + 2er
    if 2 in sorted_counts: return 1   # 2er
    
    return 0

def calculate_total_score(board):
    total = 0
    # 橫行 & 直列
    for i in range(5):
        total += get_score(board[i, :])
        total += get_score(board[:, i])
    # 對角線 (翻倍)
    total += get_score(np.diagonal(board)) * 2
    total += get_score(np.diagonal(np.fliplr(board))) * 2
    return total

# --- 蒙地卡羅 AI 核心 ---
class ScoreAI:
    def __init__(self):
        self.board = np.zeros((5, 5), dtype=int)

    def get_empty_cells(self, board):
        return list(zip(*np.where(board == 0)))

    def simulate(self, board, iterations=100):
        """隨機模擬填充剩餘空格"""
        empty_cells = self.get_empty_cells(board)
        if not empty_cells:
            return calculate_total_score(board)
        
        total_sim_score = 0
        for _ in range(iterations):
            temp_board = board.copy()
            # 隨機抽取剩餘需要的數字
            random_nums = random.choices(NUMS, weights=WEIGHTS, k=len(empty_cells))
            for i, cell in enumerate(empty_cells):
                temp_board[cell] = random_nums[i]
            total_sim_score += calculate_total_score(temp_board)
        
        return total_sim_score / iterations

    def find_best_move(self, current_num):
        empty_cells = self.get_empty_cells(self.board)
        if len(empty_cells) == 1:
            return empty_cells[0]

        best_score = -1
        best_pos = None

        print(f"正在計算數字 {current_num} 的最佳位置...", end="", flush=True)
        for cell in empty_cells:
            self.board[cell] = current_num
            # 模擬次數可調整，次數越多越準但越慢
            sim_score = self.simulate(self.board, iterations=200) 
            self.board[cell] = 0 # 恢復原狀
            
            if sim_score > best_score:
                best_score = sim_score
                best_pos = cell
            print(".", end="", flush=True)
        print("\n")
        return best_pos

# --- 執行迴圈 ---
def play_game():
    ai = ScoreAI()
    for step in range(25):
        print("\n" + "="*20)
        print(f"當前盤面 (剩餘 {25-step} 格):")
        print(ai.board)
        
        try:
            num = int(input(f"\n請輸入擲出的數字 (2-12): "))
            if num < 2 or num > 12: raise ValueError
        except ValueError:
            print("無效輸入，請輸入 2 到 12 之間的數字。")
            continue

        best_pos = ai.find_best_move(num)
        ai.board[best_pos] = num
        print(f"AI 建議放置在座標: {best_pos} (Row, Col)")

    print("\n遊戲結束！最終盤面：")
    print(ai.board)
    print(f"總分: {calculate_total_score(ai.board)}")

if __name__ == "__main__":
    play_game()