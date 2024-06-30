import torch
import numpy as np

def evaluate_agent(env, black_player, white_player, num_episodes=100):
    total_rewards = []
    win_count = {"black": 0, "white": 0, "draw": 0}
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        done = False
        ep_rewards = []

        while not info['end of game']:
            player = env.player_to_play

            if player == 0:
                action, _ = black_player.get_action(torch.tensor(obs).unsqueeze(0), eps=0.)  # Minimal exploration
            else:
                action, _ = white_player.get_action(torch.tensor(obs).unsqueeze(0), eps=0.)  # Minimal exploration
            
            obs, reward, done, info = env.step(action)
            ep_rewards.append(reward)
        ep_reward = sum(ep_rewards)
        total_rewards.append(ep_reward)
        if ep_reward > 0:
            win_count["black"] += 1
        elif ep_reward < 0:
            win_count["white"] += 1
        elif ep_reward == 0:
            win_count["draw"] += 1

    avg_reward = np.mean(total_rewards)
    black_win_rate = win_count["black"] / num_episodes
    white_win_rate = win_count["white"] / num_episodes
    draw_rate = win_count["draw"] / num_episodes

    return avg_reward, black_win_rate, white_win_rate, draw_rate