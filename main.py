import time
from agent import Agent_DQN, Agent_Random
from env import Strands_GymEnv
from stable_baselines3.common.buffers import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
from train import train_agent
from eval import evaluate_agent

def linear_schedule(start_e: float, end_e: float, duration: int, step: int) -> float:
    """
    Linear schedule for exploration decay
    """
    slope = (start_e - end_e) / duration
    return max(end_e, start_e - slope * step)

def main(size=7):
    """
    Main function to train and evaluate two agents against each other.

    Args:
        size (int, optional): The size of the game board. Defaults to 7.
    """
    # Initialize the agents
    black_player = Agent_DQN(size)
    white_player = Agent_Random(size)
    agents = [black_player, white_player]
    optimizers = [None, None]
    for i in range(2):
        if agents[i].trainable:
            optimizers[i] = torch.optim.Adam(agents[i].parameters(), lr=3e-4, eps=1e-5)

    # Initialize the environment
    env = Strands_GymEnv(size=size)

    # Initialize the training parameters
    total_timesteps = 10_000
    start_e, end_e, exploration_fraction = 1, 0.01, 0.5
    log_frequency = 100
    buffer_size = 5000
    batch_size = 256
    gamma = 1  # Discount factor for future rewards
    learning_starts = 100
    train_frequency = 10
    render_frequency = 1000

    # Initialize the replay buffers
    rb_black = ReplayBuffer(
        buffer_size, env.single_observation_space, env.single_action_space, handle_timeout_termination=False
    )
    rb_white = ReplayBuffer(
        buffer_size, env.single_observation_space, env.single_action_space, handle_timeout_termination=False
    )
    replay_buffers = [rb_black, rb_white]

    # Initialize the logs
    start_time = time.time()
    seed = 0
    logs = {"rewards": [], "ep_time": []}

    # Reset the environment and start the game
    obs, info = env.reset(seed=seed)
    states_per_game = env.max_rounds

    # Main training loop
    for global_step in range(total_timesteps):
        start_time = time.time()
        eps = linear_schedule(start_e, end_e, duration=exploration_fraction * total_timesteps, step=global_step)
        # Reset the environment for a new game
        obs, info = env.reset(seed=seed)
        ep_rewards = []

        while not info["end of game"]:
            # Determine the current player and their action
            player = env.player_to_play

            action, _ = agents[player].get_action(torch.tensor(obs).unsqueeze(0), eps)

            assert env.is_legal(action) == 1, "illegal action"

            # Execute the action
            next_obs, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            # Log the experience in the respective replay buffer

            replay_buffers[player].add(
                obs=obs, next_obs=next_obs, action=action, reward=(1 if reward > 0 else -1) * reward, done=int(done), infos=info
            )

            # Move to the next state
            obs = next_obs
            if global_step % render_frequency == 0 and global_step > learning_starts:
                env.render()
        
        if global_step % render_frequency == 0 and global_step > learning_starts:
            env.close()


        # After the game, update both agents
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                for i in range(0, 2):
                    if agents[i].trainable:
                        train_agent(agents[i], optimizers[i], replay_buffers[i], batch_size, gamma)

        end_time = time.time()
        logs["rewards"].append(sum(ep_rewards))
        logs["ep_time"].append(end_time - start_time)

        if global_step % log_frequency == 0 and global_step > learning_starts:

            env.close()
            # Evaluate the agent's performance (non-greedy)
            avg_reward = np.mean(logs["rewards"][max(0, len(logs["rewards"]) - log_frequency)::])
            black_win_rate = np.mean(
                np.sign(logs["rewards"][max(0, len(logs["rewards"]) - log_frequency)::]) == 1
            )
            draw_rate = np.mean(np.sign(logs["rewards"][max(0, len(logs["rewards"]) - log_frequency)::]) == 0)
            sps = states_per_game / np.mean(logs["ep_time"][max(0, len(logs["ep_time"]) - log_frequency)::])
            white_win_rate = 1 - black_win_rate - draw_rate
            
            print("-" * 80)
            print(f"Game {global_step}: Average Reward: {avg_reward:.2f}, Black WR: {black_win_rate:.2f}, White WR: {white_win_rate:.2f}, DR: {draw_rate:.2f}, SPS: {sps:.2f}, epsilon {eps:.2f}")
            

            # Evaluate the agent's performance (greedy)
            avg_reward, black_win_rate, white_win_rate, draw_rate = evaluate_agent(env, agents[0], agents[1], num_episodes=100)
            print("-" * 80)
            print(f"Evaluation: Average Reward: {avg_reward:.2f}, Black WR: {black_win_rate:.2f}, White WR: {white_win_rate:.2f}, DR: {draw_rate:.2f}")


        # After the game, update both agents
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                for i in range(0,2):
                    if agents[i].trainable:
                        train_agent(agents[i], optimizers[i], replay_buffers[i], batch_size, gamma)

    env.close()

if __name__ == "__main__":
    main(size=7)