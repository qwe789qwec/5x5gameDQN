import random
import numpy as np
from collections import Counter
from itertools import permutations

def get_masked_permutations(numbers, mask = [0,0,0,0,0], remove_symmetry=True):
    """
    給定一組數字與遮罩，找出固定指定位置後，剩餘數字的所有不重複排列。
    
    :param numbers: list of ints (例如 [7, 7, 8, 8, 9])
    :param mask: list of ints (0 或 1)，1 代表固定，0 代表可互換 (例如 [1, 1, 0, 1, 0])
    :param remove_symmetry: bool, 是否將「正反對稱(翻轉)」視為同一種
    """
    fixed_positions = {}  # 記錄固定數字的 {索引: 數值}
    free_numbers = []     # 記錄可以互換的數字
    free_indices = []     # 記錄可以互換的空位索引
    
    # 1. 拆分「固定數字」與「可變數字」
    for i in range(len(numbers)):
        if mask[i] == 1:
            fixed_positions[i] = numbers[i]
        else:
            free_numbers.append(numbers[i])
            free_indices.append(i)
            
    # 2. 取得「可變數字」的所有不重複排列
    unique_free_perms = set(permutations(free_numbers))
    
    all_combined_perms = set()
    
    # 3. 將這些排列塞回原本的空位中
    for perm in unique_free_perms:
        new_arr = [0] * len(numbers) # 建立空陣列
        
        # 填入固定數字
        for idx, val in fixed_positions.items():
            new_arr[idx] = val
            
        # 填入這次排列的變動數字
        for idx, val in zip(free_indices, perm):
            new_arr[idx] = val
            
        all_combined_perms.add(tuple(new_arr))

    # 4. (可選) 消除對稱 (如果這條線的正反向計分意義相同)
    if not remove_symmetry:
        return sorted(list(all_combined_perms))
        
    final_perms = set()
    for p in all_combined_perms:
        rev_p = tuple(reversed(p))
        final_perms.add(min(p, rev_p))
        
    return sorted(list(final_perms))

def check_diagonal(board):
    # return position and number of fixed values in diagonals
    diagonal_positions = [[0,0], [1,1], [2,2], [3,3], [4,4], [0,4], [1,3], [3,1], [4,0]]
    for pos in diagonal_positions:
        if board[pos[0], pos[1]] != 0:
            print(f"Diagonal position {pos} is fixed with value {board[pos[0], pos[1]]}")
    return

board = np.zeros((5, 5), dtype=int)
fix_board = None

dice_array = np.random.choice(range(2, 13), size=25, p=[1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])
print("骰子數值:", dice_array)
counts = Counter(dice_array)
print("骰子分布:", counts)

straights = [
    [2,3,4,5,6],
    [3,4,5,6,7],
    [4,5,6,7,8],
    [5,6,7,8,9],
    [6,7,8,9,10],
    [7,8,9,10,11],
    [8,9,10,11,12],
]


possible_hughscores = []
for s in straights:
    if all(num in dice_array for num in s):
        possible_hughscores.append(s)

for num in counts:
    if counts[num] >= 5:
        possible_hughscores.append([int(num), int(num), int(num), int(num), int(num)])
    elif counts[num] >= 3:
        for other_num in counts:
            if counts[other_num] >= 2 and other_num != num:
                possible_hughscores.append([int(num), int(num), int(num), int(other_num), int(other_num)])

print("可能的高分組合:", possible_hughscores)

for combo in possible_hughscores:
    rest_dice = dice_array.copy()
    for num in combo:
        rest_dice = np.delete(rest_dice, np.where(rest_dice == num)[0][0])
    if fix_board 
    arrangements = get_masked_permutations(combo)

    for arr in arrangements:
        for combo2 in possible_hughscores:
            if arr[3] in combo2:
                combo2_dice = combo2.copy()
                combo2_dice = np.delete(combo2_dice, np.where(combo2_dice == arr[3])[0][0])
                for num in combo2_dice:
                    rest_dice = np.delete(rest_dice, np.where(rest_dice == num)[0][0])
                arrangements2 = get_masked_permutations(combo2_dice)
            else:
                break
            

