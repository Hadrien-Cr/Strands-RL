import torch
import torch.nn as nn
import numpy as np
import random

# Layer initialization function
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class Agent_DQN(nn.Module):
    def __init__(self,board_size):
        super().__init__()
        self.board_size = board_size
        
        self.nn = nn.Sequential(
            layer_init(nn.Linear(3*self.board_size * self.board_size, self.board_size * self.board_size,)),
            nn.ReLU(),
            layer_init(nn.Linear(self.board_size * self.board_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, self.board_size*self.board_size)),
        )
        
    def get_action(self, obs,eps = 0):
        bitmap_empty, bitmap_b, bitmap_w, mask= (
        torch.tensor([obs[i][0] for i in range(len(obs))]).float(),
        torch.tensor([obs[i][1] for i in range(len(obs))]).float(),
        torch.tensor([obs[i][2] for i in range(len(obs))]).float(),
        torch.tensor([obs[i][3] for i in range(len(obs))]).float())
        
        x = torch.concat([bitmap_empty,bitmap_b,bitmap_w],dim=-1)
        values = self.nn(x)
        values_masked = torch.mul(values, mask) -1e9 * (1 - mask)
        if random.random()<eps:
            action = (mask+0.01*torch.rand(mask.size())).argmax(dim=-1)
        else:
            action = values_masked.argmax(dim=-1)
        assert torch.any(mask[torch.arange(x.shape[0]),action]==1), "illegal action"
        return action,values

if __name__ == "__main__":
    bs = 8
    agent = Agent_DQN(3)
    bitmap_empty, bitmap_b, bitmap_w, mask = np.random.normal(0,1,(bs,9)),np.random.normal(0,1,(bs,9)),np.random.normal(0,1,(bs,9)),np.array([[1,1,0,0,0,0,0,0,0]]*bs)
    obs = [(bitmap_empty[i], bitmap_b[i], bitmap_w[i], mask[i]) for i in range(bs)]
    mask = torch.tensor([[1,1,0,0,0,0,0,0,0]]*bs)
    action,values =agent.get_action(obs,eps = 0.5)
    print(action)




def train_agent(agent, optimizer, replay_buffer, batch_size, gamma):
    """
    Train the agent using the replay buffer.
    """
    # Sample a batch from the replay buffer
    sample_size = min(batch_size,replay_buffer.size())
    obs, next_obs, actions, rewards, dones = replay_buffer.sample(sample_size)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    # Compute the current Q values
    _,q_values = agent.get_action(obs)
    
    q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    
    # Compute the target Q values
    with torch.no_grad():
        _,next_q_values = agent.get_action(next_obs)
        max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + (gamma * max_next_q_values * (1 - torch.tensor(dones,dtype = torch.float)))
    
    # Compute the loss
    loss = nn.MSELoss()(q_values, target_q_values)
    # Optimize the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class ReplayBuffer:
    def __init__(self, buffer_size,  seed=None):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
    def add(self, state, next_state, action, reward, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)  # Expand the buffer size

        self.buffer[self.position] = (state, next_state, action, reward, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states =  [batch[i][0] for i in range(batch_size)]
        next_states = [batch[i][1] for i in range(batch_size)]
        actions = [batch[i][2] for i in range(batch_size)]
        rewards = [batch[i][3] for i in range(batch_size)]
        dones = [batch[i][4] for i in range(batch_size)]
        return (states, next_states, actions, rewards, dones)
    
    def size(self):
        return len(self.buffer)
