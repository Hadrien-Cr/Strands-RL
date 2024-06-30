import torch
import torch.nn as nn
import numpy as np
import random
import time

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
        self.trainable = True
        self.nn = nn.Sequential(
            layer_init(nn.Linear(4*self.board_size * self.board_size, self.board_size * self.board_size,)),
            nn.ReLU(),
            layer_init(nn.Linear(self.board_size * self.board_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, self.board_size*self.board_size)),
        )
        
    def get_action(self, obs,eps = 0):
        mask = obs[:,3*self.board_size* self.board_size:]
        values = self.nn(obs.float())
        values_masked = torch.mul(values, mask) - 1e20 * (1 - mask)
        
        if random.random()<eps:
            action = (mask+0.01*torch.rand(mask.size())).argmax(dim=-1)
        else:
            action = values_masked.argmax(dim=-1)
        assert torch.any(mask[torch.arange(obs.size()[0]),action]==1), "illegal action"

        return action,values
    
class Agent_Random(nn.Module):
    def __init__(self,board_size):
        super().__init__()
        self.board_size = board_size
        self.trainable = False

    def get_action(self, obs,eps = 0):
        mask = obs[:,3*self.board_size* self.board_size:]
        values = torch.rand(obs.size()[0],self.board_size*self.board_size)
        values_masked =  torch.mul(values, mask)  - 1e20 * (1 - mask)
        
        if random.random()<eps:
            action = (mask+0.01*torch.rand(mask.size())).argmax(dim=-1)
        else:
            action = values_masked.argmax(dim=-1)
        assert torch.any(mask[torch.arange(obs.size()[0]),action]==1), "illegal action"
        
        return action,values

if __name__ == "__main__":
    from gymnasium.spaces import MultiDiscrete
    bs = 8
    size = 7
    agent = Agent_DQN(size)
    observation_space = MultiDiscrete([2]*(4*size*size), seed=42)
    obs = torch.tensor([observation_space.sample() for _ in range(bs)])
    actions,q_values = agent.get_action(obs,eps = 0.5)
    print(obs.shape,actions.size(),q_values.size())
    q_values = torch.gather(q_values,1, actions.unsqueeze(-1)).squeeze(-1)
    print(obs.shape,actions.size(),q_values.size())


