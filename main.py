from agent import Agent_TD, Agent_Random
from eval import evaluate_agent, rollout_and_render
from env import Strands_GymEnv
import torch
import torch.nn as nn
import random
import time
import numpy as np
from itertools import count

class BaseModel(nn.Module):
    def __init__(self, size = 11,hidden_units = 128, lr = 0.001, lamda = 0.5, seed=0):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.lamda = lamda  # trace-decay parameter
        self.start_episode = 0
        self.board_size = size

        self.eligibility_traces = None
        self.optimizer = None

        torch.manual_seed(seed)
        random.seed(seed)
        input
        self.hidden = nn.Sequential(
            nn.Linear(4*self.board_size*self.board_size, hidden_units),
            nn.Sigmoid())

        self.output = nn.Sequential(
            nn.Linear(hidden_units, 1),
            nn.Sigmoid()).float()

        for p in self.parameters():
            nn.init.zeros_(p)

    def forward(self, obs):
        x = torch.from_numpy(obs).to(torch.float32)
        x = self.hidden(x)
        x = self.output(x)
        return x

    def update_weights(self, p, p_next):
        # reset the gradients
        self.zero_grad()

        # compute the derivative of p w.r.t. the parameters
        p.backward()

        with torch.no_grad():

            td_error = p_next - p

            # get the parameters of the model
            parameters = list(self.parameters())

            for i, weights in enumerate(parameters):

                # z <- gamma * lambda * z + (grad w w.r.t P_t)
                self.eligibility_traces[i] = self.lamda * self.eligibility_traces[i] + weights.grad

                # w <- w + alpha * td_error * z
                new_weights = weights + self.lr * td_error * self.eligibility_traces[i]
                weights.copy_(new_weights)

        return td_error

    def init_eligibility_traces(self):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False) for weights in list(self.parameters())]

    def train_agent(self, n_episodes = 10_000, eligibility=False):
        
        # INITIALIZATION OF THE AGENTS
        network = self
        agents = [ Agent_TD('black', board_size=self.board_size, net=network),  Agent_TD('white', board_size=self.board_size, net=network) ]
        agent_random = Agent_Random('white', board_size=self.board_size, net=network)

        # METRICS AND LOGS
        wins = {'white': 0, 'black': 0, 'draw':0}
        rewards = []
        logs_sps = []
        logs_loss =  []
        n_eval = 100
        freq_eval,freq_render, freq_log = 1000,1000,50

        env = Strands_GymEnv(self.board_size)
        start_eps, end_eps, exploration_fraction = 0.5, 0.01, 0.5
        
        for episode in range(1,n_episodes+1):
            eps = max(start_eps - episode * exploration_fraction/n_episodes  , end_eps) 
            if eligibility:
                self.init_eligibility_traces()

            obs, info = env.reset()
        
            st = time.time()

            for i in count():
                agent = agents[env.player_to_play]
                p = self(obs)
                action = agent.choose_best_action(env,eps = eps)
                obs_next, reward, done, info = env.step(action)
                p_next = self(obs_next)

                if done:
                    loss = self.update_weights(p, reward)
                    if reward != 0:
                        winner = ('white' if reward < 0 else 'black')
                        wins[winner] += 1
                    else:
                        wins['draw'] += 1
                    rewards.append(reward)
                    dt = time.time() - st
                    logs_sps.append(i/dt)
                    break
                else:
                    if episode>=env.max_rounds-i-1:
                        loss = self.update_weights(p, p_next)

                obs = obs

            if episode % freq_log == 0:
                avg_reward, black_win_rate, white_win_rate, draw_rate,sps = np.mean(rewards), wins['black']/episode, wins['white']/episode, wins['draw']/episode,np.mean(logs_sps)
                print(f"Training:  {episode} games played, Avg reward = {avg_reward:.3f}, Black win rate = {black_win_rate:.3f}, White win rate = {white_win_rate:.3f}, Draw rate = {draw_rate:.3f}, SPS = {round(sps)}") 

            if episode % freq_eval == 0:
                print('-'*110)
                avg_reward, black_win_rate, white_win_rate, draw_rate,sps = evaluate_agent( env, [agents[0],agent_random], num_episodes=n_eval)
                print(f"Evaluation vs Random: Avg reward = {avg_reward:.3f}, Black win rate = {black_win_rate:.3f}, White win rate = {white_win_rate:.3f}, Draw rate = {draw_rate:.3f} SPS = {round(sps)}") 
                print('-'*110)

            if episode % freq_render == 0:
                rollout_and_render(env,agents = [agents[0],agent_random])

model = BaseModel()
model.train_agent(n_episodes = 10_000, eligibility=True)