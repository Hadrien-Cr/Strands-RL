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
    def __init__(self,nbHexes: int, nbDigits: int, LABEL_COLOR: int, is_random_agent: bool = False, device: str = "cpu") -> None:
        super().__init__()

        self.LABEL = LABEL_COLOR 

        self.fc1 = nn.Linear(nbHexes,128)
        self.fc2 = nn.Linear(128,128)

        self.outDigits = nn.Linear(128,nbDigits)
        self.outHexes = nn.Linear(128,nbHexes)
        
        self.is_random_agent = is_random_agent
        self.nbHexes = nbHexes
        self.nbDigits = nbDigits

        self.device = device

        self.to(device)

    def get_activations_digits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).to(self.device)
        if self.is_random_agent:
            return torch.randn(self.nbDigits).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.outDigits(x))
        return x.cpu()
    
    def get_activations_hexes(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).to(self.device)
        if self.is_random_agent:
            return torch.randn(self.nbHexes).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.outHexes(x))
        return x.cpu()
    

    def act_reinforce(self, board: StrandsBoard) -> torch.Tensor:
        """
        Acts and returns the (differentiable) log probability of the picked action
        """
        sum_log_prob, n_items = torch.tensor([0.]), 0

        # choosing a digit
        if board.digits_left_to_place==0:
            x = board.compute_network_inputs()
            activations = self.get_activations_digits(x)
            mask = board.get_digits_availables()

            mask_tensor = torch.tensor(mask,dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor),dim=-1)
            m = torch.distributions.Categorical(probs)
            A = m.sample()

            board.update_digit_chosen(A.item())

            sum_log_prob += m.log_prob(A)
            n_items += 1

        # placing tiles on hexes
        while board.digits_left_to_place>0:
            x = board.compute_network_inputs()
            activations = self.get_activations_hexes(x)
            mask = board.get_hexes_availables()

            mask_tensor = torch.tensor(mask,dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor),dim=-1)
            m = torch.distributions.Categorical(probs)
            A = m.sample()

            board.update_hex(A.item(), self.LABEL)

            sum_log_prob += m.log_prob(A)
            n_items += 1

        return (sum_log_prob)


    def act_greedily(self, board: StrandsBoard) :
        """
        Acts greedily without returning anything
        """
        # choosing a digit
        if board.digits_left_to_place==0:
            x = board.compute_network_inputs()
            activations = self.get_activations_digits(x)
            mask = board.get_digits_availables()

            mask_tensor = torch.tensor(mask,dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor),dim=-1)
            A = probs.argmax(dim=-1)

            board.update_digit_chosen(A.item())

        # placing tiles on hexes
        while board.digits_left_to_place>0:
            x = board.compute_network_inputs()
            activations = self.get_activations_hexes(x)
            mask = board.get_hexes_availables()

            mask_tensor = torch.tensor(mask,dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor),dim=-1)
            A = probs.argmax(dim=-1)

            board.update_hex(A.item(), self.LABEL)


def init_agents(board: StrandsBoard, device: str = "cpu") -> list[Agent]:
    agents = [Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_WHITE, device = device),
              Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_BLACK, device = device)]
    return agents

def init_random_agents(board: StrandsBoard, device: str = "cpu") -> list[Agent]:
    agents = [Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_WHITE, device = device, is_random_agent=True),
              Agent(nbDigits=board.LABEL_BLACK+1, nbHexes=board.nbHexes, LABEL_COLOR=board.LABEL_BLACK, device = device, is_random_agent=True)]
    return agents