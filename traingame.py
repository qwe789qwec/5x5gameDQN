from dicegame import gameEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# --- 1. 神經網路模型 ---
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # 盤面 one-hot 編碼後是 5x5x13 = 325 維 (0-12 的 One-hot)
        self.fc_board1 = nn.Linear(325, 128)
        self.fc_board2 = nn.Linear(128, 128)
        
        # 盤面輸出 128 維 + 骰子 13 維 = 141 維
        self.fc_combined = nn.Linear(141, 256)
        
        # 最終輸出還是 25 個 Q 值 (對應 5x5 的 25 個格子)
        self.fc_out = nn.Linear(256, 25) 

    def forward(self, board_one_hot, dice_one_hot):
        # 處理盤面 (輸入維度: batch_size x 325)
        x = F.relu(self.fc_board1(board_one_hot))
        x = F.relu(self.fc_board2(x))
        
        # 將 13 維的骰子 One-hot 直接串接進來
        # 注意：這裡不需要再做 unsqueeze，因為 dice_one_hot 已經是 [batch_size, 13]
        x = torch.cat((x, dice_one_hot), dim=1) # 串接後變成 [batch_size, 141]
        
        x = F.relu(self.fc_combined(x))
        q_values = self.fc_out(x)
        return q_values

# --- 2. 經驗回放池 (Replay Buffer) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# --- 3. DQN 代理人 (Agent) ---
class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 64

    def format_state(self, board, dice):
        # 將盤面轉為 One-hot (5x5x13)
        board_one_hot = np.zeros((5, 5, 13), dtype=np.float32)
        for r in range(5):
            for c in range(5):
                val = board[r, c]
                board_one_hot[r, c, val] = 1.0 

        # 骰子也轉為 One-hot (13維)
        dice_one_hot = np.zeros(13, dtype=np.float32)
        dice_one_hot[dice] = 1.0

        return board_one_hot.flatten(), dice_one_hot

    def select_action(self, board, dice, empty_cells):
        if random.random() < self.epsilon:
            # 隨機探索：從「空位」中隨機選一個
            idx = random.choice(range(len(empty_cells)))
            row, col = empty_cells[idx]
            action = row * 5 + col
            return action

        with torch.no_grad():
            board_flat, dice_val = self.format_state(board, dice)
            board_t = torch.FloatTensor(board_flat).unsqueeze(0).to(self.device)
            dice_t = torch.FloatTensor(dice_val).unsqueeze(0).to(self.device)

            q_values = self.policy_net(board_t, dice_t).squeeze()
            
            # Action Masking (非常重要！把非空格的 Q 值設為極小)
            mask = torch.ones(25, dtype=torch.bool).to(self.device)
            for row, col in empty_cells:
                mask[row * 5 + col] = False
            q_values[mask] = -1e9 
            
            return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 解析狀態
        boards = torch.FloatTensor(states[:, 0:325]).to(self.device)
        dices = torch.FloatTensor(states[:, 325:]).to(self.device)
        
        next_boards = torch.FloatTensor(next_states[:, 0:325]).to(self.device)
        next_dices = torch.FloatTensor(next_states[:, 325:]).to(self.device)
        
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 計算目前的 Q 值
        curr_q = self.policy_net(boards, dices).gather(1, actions)
        
        # 計算目標 Q 值 (Double DQN 概念，這裡簡化為一般 DQN)
        with torch.no_grad():
            # 注意：在計算 next_q 時，我們無法預測未來的骰子點數。
            # 為了簡化，如果遊戲沒結束，我們取未來所有可能行動的最大 Q 值。
            # 更嚴謹的做法應該是計算未來骰子點數的期望值 (Expected Sarsa)，但一般 DQN 這裡先取 max。
            next_q = self.target_net(next_boards, next_dices).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = F.mse_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay