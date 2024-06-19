import gymnasium as gym

def fixed_strategy(observation):
    # observation[2] 是杆子的角度
    angle = observation[2]
    return 0 if angle < 0 else 1

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):  # 增加迭代次數以延長測試時間
    action = fixed_strategy(observation)  # 使用固定策略決定動作
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()  # 渲染環境以視覺化測試過程

    if terminated or truncated:
        observation, info = env.reset()
        print('Episode ended, resetting environment')

env.close()

