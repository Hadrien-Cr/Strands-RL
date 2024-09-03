import torch 
import torch.nn as nn
import torch.nn.functional as F
from env import *
import random
def random_draw_from_boolean_list(l:list[bool])-> int:
    valid_indices = []
    for i in range(len(l)):
        if l[i]:
            valid_indices.append(i)
    return random.choice(valid_indices)

class Agent(nn.Module):
    def __init__(self,nbHexes: int,nbDigits: int, LABEL_COLOR: int, is_random_agent: bool = False) -> None:
        super().__init__()

        self.LABEL = LABEL_COLOR 

        self.fc1 = nn.Linear(nbHexes,128)
        self.fc2 = nn.Linear(128,128)

        self.outDigits = nn.Linear(128,nbDigits)
        self.outHexes = nn.Linear(128,nbHexes)
        
        self.is_random_agent = is_random_agent
        self.nbHexes = nbHexes
        self.nbDigits = nbDigits

    def get_activations_digits(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_random_agent:
            return torch.randn(self.nbDigits)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.outDigits(x))
        return x
    
    def get_activations_hexes(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_random_agent:
            return torch.randn(self.nbHexes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.outHexes(x))
        return x
    

def act(agent: Agent, board: StrandsBoard, eps: float = 0) -> torch.Tensor:
    """
    Acts and returns the surrogate Q value of the action performed
    """
    sum_Q, n_items = torch.tensor(0.), 0

    # choosing a digit
    if board.digits_left_to_place==0:
        x = board.compute_network_inputs()
        activations = agent.get_activations_digits(x)
        mask = board.get_digits_availables()

        mask_tensor = torch.tensor(mask,dtype=torch.float)
        activations = torch.softmax(activations + torch.log(mask_tensor) ,dim=0)

        A = torch.argmax(activations).item()
        Q = activations[A]
        if not mask[A]:
            print(activations,mask)
        assert mask[A], "wrong action"
        board.update_digit_chosen(A)

        if random.random() < eps:
            A = random_draw_from_boolean_list(mask)

        board.update_digit_chosen(A)

        sum_Q = sum_Q + Q
        n_items += 1

    # placing tiles on hexes
    while board.digits_left_to_place>0:
        x = board.compute_network_inputs()
        activations = agent.get_activations_hexes(x)
        mask = board.get_hexes_availables()

        mask_tensor = torch.tensor(mask,dtype=torch.float)
        activations = torch.softmax(activations + torch.log(mask_tensor) ,dim=0)

        A = torch.argmax(activations).item()
        Q = activations[A]
        if not mask[A]:
            print(activations,mask)
        
        if random.random() < eps:
            A = random_draw_from_boolean_list(mask)
        assert mask[A], "wrong action"
        board.update_hex(A, agent.LABEL)

        sum_Q = sum_Q + Q
        n_items += 1

    return (sum_Q/n_items)



def init_agents(board: StrandsBoard) -> list[Agent]:
    agents = [Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_WHITE),
              Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_BLACK)]
    return agents

def init_random_agents(board: StrandsBoard) -> list[Agent]:
    agents = [Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_WHITE, is_random_agent=True),
              Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_BLACK, is_random_agent=True)]
    return agents