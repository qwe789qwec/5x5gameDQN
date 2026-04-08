from dicegame import Game
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Environment Wrapper
# =========================
class DiceGameEnv:
    def __init__(self):
        self.game = Game()
        self.current_number = None

    def reset(self):
        self.game.reset_game()
        self.current_number = self.game.roll_dice()
        return self.get_state()

    def get_state(self):
        """
        state:
        - board: 5x5
        - current dice number: 1 scalar
        合併成 26 維向量
        """
        board_flat = self.game.board.flatten().astype(np.float32)
        number = np.array([self.current_number], dtype=np.float32)
        state = np.concatenate([board_flat, number])
        return state

    def get_valid_actions(self):
        """
        回傳可下的位置 index (0~24)
        """
        empty_cells = self.game.get_empty_cells()
        return [r * 5 + c for r, c in empty_cells]

    def step(self, action):
        """
        action: 0~24
        """
        row = action // 5
        col = action % 5

        old_score = self.game.calculate_total_score()
        success = self.game.place_number(row, col, self.current_number)

        if not success:
            # 非法動作直接給懲罰，並結束這回合 or 不結束都可以
            reward = -5.0
            done = True
            return self.get_state(), reward, done

        new_score = self.game.calculate_total_score()
        reward = float(new_score - old_score)

        done = self.game.end_game()

        if not done:
            self.current_number = self.game.roll_dice()

        return self.get_state(), reward, done


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# Q Network
# =========================
class DQN(nn.Module):
    def __init__(self, input_dim=26, output_dim=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Agent
# =========================
class DQNAgent:
    def __init__(self, device="cpu"):
        self.device = device

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(50000)

        self.gamma = 0.99
        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state, valid_actions):
        """
        epsilon-greedy + mask 無效動作
        """
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()

        masked_q = np.full_like(q_values, -1e9, dtype=np.float32)
        masked_q[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked_q))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =========================
# Training Loop
# =========================
def train(num_episodes=3000, target_update=50, save_path="dqn_dicegame.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env = DiceGameEnv()
    agent = DQNAgent(device=device)

    reward_history = []
    score_history = []
    loss_history = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            next_state, reward, done = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        final_score = env.game.calculate_total_score()
        reward_history.append(total_reward)
        score_history.append(final_score)

        if losses:
            loss_history.append(np.mean(losses))
        else:
            loss_history.append(0.0)

        agent.decay_epsilon()

        if episode % target_update == 0:
            agent.update_target()

        if episode % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            avg_score = np.mean(score_history[-100:])
            avg_loss = np.mean(loss_history[-100:])
            print(
                f"Episode {episode:4d} | "
                f"Avg Reward: {avg_reward:6.2f} | "
                f"Avg Score: {avg_score:6.2f} | "
                f"Avg Loss: {avg_loss:8.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# =========================
# Evaluation
# =========================
def evaluate(model_path="dqn_dicegame.pth", num_games=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = DiceGameEnv()
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores = []

    for game_idx in range(num_games):
        state = env.reset()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor).squeeze(0).cpu().numpy()

            masked_q = np.full_like(q_values, -1e9, dtype=np.float32)
            masked_q[valid_actions] = q_values[valid_actions]
            action = int(np.argmax(masked_q))

            state, reward, done = env.step(action)

        final_score = env.game.calculate_total_score()
        scores.append(final_score)

        print(f"Game {game_idx + 1}: score = {final_score}")
        env.game.display_board()
        print("-" * 30)

    print("Average score:", np.mean(scores))


if __name__ == "__main__":
    train(num_episodes=3000)
    evaluate(num_games=5)