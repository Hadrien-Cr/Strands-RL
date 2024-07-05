import torch
import numpy as np
from itertools import count
import time

def evaluate_agent(env, agents, num_episodes=100):

    wins = {"black": 0, "white": 0, "draw": 0}
    rewards = []
    logs_sps = []
    with torch.no_grad():
        for episode in range(num_episodes):

            obs, info = env.reset(seed=episode)
            done = False
            st = time.time()
            for i in count():
                player = env.player_to_play
                action = agents[player].choose_best_action(env, eps=0.)  # Minimal exploration
                obs, reward, done, info = env.step(action)
            
                if done:
                    if reward > 0:
                        wins["black"] += 1
                    elif reward < 0:
                        wins["white"] += 1
                    elif reward == 0:
                        wins["draw"] += 1
                    dt = time.time() - st
                    logs_sps.append(i/dt)
                    rewards.append(reward)
                    break
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    black_win_rate = wins["black"] / num_episodes
    white_win_rate = wins["white"] / num_episodes
    draw_rate = wins["draw"] / num_episodes
    sps = np.mean(logs_sps)

    return avg_reward, std_reward, black_win_rate, white_win_rate, draw_rate,sps

def rollout_and_render(env, agents):
    obs, info = env.reset()  # Reset environment
    done = False
    env.render()
    while not done:
        player = env.player_to_play
        action = agents[player].choose_best_action(env, eps=0.)  # Minimal exploration
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()