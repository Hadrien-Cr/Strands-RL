import time
from agent import Agent_DQN,train_agent,ReplayBuffer
from env import Strands_GymEnv
import torch
import torch.nn as nn
import numpy as np

def linear_schedule(start_e: float, end_e: float, duration: int, step: int) -> float:
    """
    Linear schedule for exploration decay
    """
    slope = (start_e - end_e) / duration
    return max(end_e, start_e - slope * step)

def main(size=7):
    black_player = Agent_DQN(size)
    white_player = Agent_DQN(size)
    optimizer_white = torch.optim.Adam(white_player.parameters(), lr=1e-3)
    optimizer_black = torch.optim.Adam(black_player.parameters(), lr=1e-1)

    env = Strands_GymEnv(size=size)
    total_timesteps = 1000
    freq_render = 100
    buffer_size = 1000
    batch_size = 64
    gamma = 0.99  # Discount factor for future rewards

    rb_black = ReplayBuffer(buffer_size)
    
    rb_white = ReplayBuffer(buffer_size)

    start_time = time.time()
    start_e, end_e, exploration_fraction = 1.0, 0.01, 0.5
    seed = 0
    logs = {'rewards': []}
    for global_step in range(total_timesteps):
        eps = linear_schedule(start_e, end_e, exploration_fraction, global_step)
        # Set up environment and reset for new game
        obs = env.reset(seed=seed)
        done = False
        ep_rewards = []

        while not done:
            # Determine the current player and their action
            player = env.player_to_play
            
            if player == 0:
                action, _ = black_player.get_action([obs],eps)
            else:
                action, _ = white_player.get_action([obs],eps)
            
            assert env.current_mask[action] == 1, "illegal action"

            # Execute the action
            next_obs, reward, done, info = env.step(action)
            ep_rewards.append(reward)

            # Log the experience in the respective replay buffer
            if player == 0:
                rb_black.add(obs, next_obs, action, reward, done)
            else:
                rb_white.add(obs, next_obs, action, -reward, done)
            
            # Move to the next state
            obs = next_obs

            if global_step % freq_render == 0:
                env.render()

        # After the game, update both agents
        for _ in range(10):  # Let's update the network multiple times after each game
            if rb_black.size() > batch_size:
                train_agent(black_player, optimizer_black, rb_black, batch_size, gamma)

            # if rb_white.size() > batch_size:
            #     train_agent(white_player, optimizer_white, rb_white, batch_size, gamma)

        logs['rewards'].append(sum(ep_rewards))
        running_avg = np.mean(logs['rewards'][min(0,len(logs['rewards'])-100)::])
        running_wr = np.mean(np.sign(logs['rewards'][min(0,len(logs['rewards'])-100)::])+1)/2
        print(f"Game {global_step+1}:  Reward {sum(ep_rewards):.2f}, \
                Average Reward: {np.mean(logs['rewards'][-100::]):.2f},  \
                Average WR: {running_wr:.2f}")
    env.close()

if __name__ == "__main__":
    main()
