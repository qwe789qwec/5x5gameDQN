import torch
from traingame import DQNAgent
from dicegame import gameEnv
import numpy as np

def train_dqn(episodes=5000, model_save_path=None):
    env = gameEnv()
    agent = DQNAgent() # 使用上一篇提供的 DQN Agent

    if model_save_path is not None:
        print(f"正在載入模型權重: {model_save_path}")
        agent.policy_net.load_state_dict(torch.load(model_save_path, map_location=agent.device))
        agent.policy_net.eval()
        print("模型權重載入完成！")
    
    # 記錄分數用
    scores_history = []
    
    for e in range(episodes):
        # 1. 初始化環境
        board, dice = env.reset()
        state_board, state_dice = agent.format_state(board, dice)
        state_combined = np.concatenate([state_board, state_dice])
        
        done = False
        
        while not done:
            # 2. 取得合法動作名單 (Action Masking 需要)
            # 將 0-24 反向轉換為 (row, col) 格式給 Agent 使用
            valid_actions = env.get_valid_actions()
            
            # 3. 代理人選擇動作 (輸入目前的盤面、骰子、可選空位)
            action_idx = agent.select_action(board, dice, valid_actions)
            
            # 4. 執行動作 (傳遞 0-24 的整數給 env)
            (next_board, next_dice), reward, done = env.step(action_idx)
            
            # 5. 格式化 Next State
            next_state_board, next_state_dice = agent.format_state(next_board, next_dice)
            next_state_combined = np.concatenate([next_state_board, next_state_dice])
            
            # 6. 存入記憶體並訓練
            agent.memory.push(state_combined, action_idx, reward, next_state_combined, done)
            agent.train_step()
            
            # 7. 狀態更新
            board, dice = next_board, next_dice
            state_combined = next_state_combined
            
        # 回合結束更新 Epsilon (降低隨機探索率)
        agent.update_epsilon()
        scores_history.append(env.calculate_total_score())
        
        # 定期更新 Target Network
        if e % 10 == 0:
            agent.update_target_network()
            
        # 打印進度
        if e % 100 == 0 and e > 0:
            avg_score = np.mean(scores_history[-100:])
            print(f"Episode: {e}/{episodes} | Avg Score (last 100): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    return agent, scores_history

def test_ai_manually(model_path):
    print("\n=== 載入 AI 模型中 ===")
    
    # 1. 初始化一個全新的 Agent
    agent = DQNAgent()
    
    # 2. 載入剛剛存檔的權重
    # 如果你的電腦沒有 GPU，加入 map_location=torch.device('cpu') 確保載入順利
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    
    # 3. 將網路設定為推論模式 (Evaluation Mode)，並關閉隨機探索
    agent.policy_net.eval()
    agent.epsilon = 0.0  # 強制 AI 只做最好的決定，不隨機亂走
    
    # 4. 初始化底層遊戲環境
    game = gameEnv()
    game.reset()
    
    print("=== AI 測試開始 ===")
    while not game.end_game():
        # 顯示目前盤面
        print("\n目前盤面：")
        print(game.board)
        
        # 讓玩家手動輸入骰子點數
        while True:
            try:
                dice_input = int(input("🎲 請輸入這回合的骰子點數 (2-12): "))
                if 2 <= dice_input <= 12:
                    break
                else:
                    print("⚠️ 點數必須在 2 到 12 之間！")
            except ValueError:
                print("⚠️ 請輸入有效的數字！")
                
        
        # AI 根據盤面與你輸入的點數，選擇動作
        action_idx = agent.select_action(game.board, dice_input, game.get_valid_actions())
        
        # 將 1D 的 action_idx 轉回 2D 的 (row, col)
        row, col = action_idx // 5, action_idx % 5
        print(f"🤖 AI 決定將 {dice_input} 放在座標: ({row}, {col})")
        
        # 執行動作
        game.place_number(row, col, dice_input)

    # 遊戲結束結算
    print("\n=== 🏁 遊戲結束 ===")
    print("最終盤面：")
    print(game.board)
    final_score = game.calculate_total_score()
    print(f"🏆 AI 最終總分: {final_score}")

# 執行手動測試 (請確保已經執行過訓練並產生了 pth 檔案)
test_ai_manually("dqn_knister_fullscore.pth")

# trained_agent, history = train_dqn(episodes=5000)
# torch.save(trained_agent.policy_net.state_dict(), "dqn_knister_fullscore.pth")
# print("✅ 模型已成功儲存至 dqn_knister_fullscore.pth")