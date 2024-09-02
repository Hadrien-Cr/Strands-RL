import torch 
import torch.nn as nn
import torch.nn.functional as F
from env import *

class Agent(nn.Module):
    def __init__(self,nbHexes: int,nbDigits: int, LABEL_COLOR: int):
        super().__init__()

        self.LABEL = LABEL_COLOR 

        self.fc1 = nn.Linear(nbHexes,128)
        self.fc2 = nn.Linear(128,128)

        self.outDigits = nn.Linear(128,nbDigits)
        self.outHexes = nn.Linear(128,nbHexes)
        

    def get_activations_digits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.outDigits(x)
        return x
    
    def get_activations_hexes(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.outHexes(x)
        return x
    

def act(agent: Agent, board: StrandsBoard) -> torch.Tensor:
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
        activations = activations - (1-mask_tensor) * 1e10

        A = torch.argmax(activations).item()
        Q = activations[A]
        board.update_digit_chosen(A)

        sum_Q = sum_Q + Q
        n_items += 1

    # placing tiles on hexes
    while board.digits_left_to_place>0:
        x = board.compute_network_inputs()
        activations = agent.get_activations_hexes(x)
        mask = board.get_hexes_availables()

        mask_tensor = torch.tensor(mask,dtype=torch.float)
        activations = activations - (1-mask_tensor) * 1e10

        A = torch.argmax(activations).item()
        Q = activations[A]

        board.update_hex(A, agent.LABEL)

        sum_Q = sum_Q + Q
        n_items += 1

    return (sum_Q/n_items)


def init_agents(board: StrandsBoard) -> list[Agent]:
    agents = [Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_WHITE),
              Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_BLACK)]
    return agents