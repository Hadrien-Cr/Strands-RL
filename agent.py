import torch
import torch.nn as nn
import numpy as np
import random
import time



class Agent_TD:
    def __init__(self,color,board_size,net):
        super().__init__()
        self.color = color
        self.board_size = board_size
        self.net = net
        
    def choose_best_action(self, env,eps = 0):
        actions = env.list_available_actions()
        
        if random.random()<eps:
            return(random.choice(actions))
        
        else:
            values = [0.0] * len(actions)
            env.store_state()
            for i, action in enumerate(actions):
                obs, reward, done, info = env.step(action, restore_after_call = True)
                values[i] = self.net(obs)

        best_action_index = torch.argmax(torch.tensor(values)) if self.color == 'black' else torch.argmin(torch.tensor(values))
        best_action = list(actions)[best_action_index]

        return best_action


class Agent_Random:
    def __init__(self,color,board_size,net):
        super().__init__()
        self.color = color
        self.board_size = board_size
        
    def choose_best_action(self, env,eps = 0):
        actions = env.list_available_actions()
        return(random.choice(actions))
        


