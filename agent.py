import torch
import torch.nn as nn
import torch.nn.functional as F
from env import *
import random

class Agent(nn.Module):
    def __init__(self, nbHexes: int, nbDigits: int, LABEL_COLOR: int, device: str = "cpu") -> None:
        super().__init__()

        self.LABEL = LABEL_COLOR
        self.nbHexes = nbHexes
        self.nbDigits = nbDigits
        self.device = device

        self.to(device)

    def get_activations_digits(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_activations_hexes(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def act_reinforce(self, board: StrandsBoard) -> torch.Tensor:
        """
        Acts and returns the (differentiable) log probability of the picked action.
        """
        log_prob_digit, log_prob_hex, n_items = torch.tensor([0.]), torch.tensor([0.]), 0

        # choosing a digit
        if board.digits_left_to_place == 0:
            x = board.compute_network_inputs()
            activations = self.get_activations_digits(x)
            mask = board.get_digits_availables()

            mask_tensor = torch.tensor(mask, dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor), dim=-1)
            m = torch.distributions.Categorical(probs)
            A = m.sample()

            board.update_digit_chosen(A.item())

            log_prob_digit = m.log_prob(A)

        # placing tiles on hexes
        while board.digits_left_to_place > 0:
            x = board.compute_network_inputs()
            activations = self.get_activations_hexes(x)
            mask = board.get_hexes_availables()

            mask_tensor = torch.tensor(mask, dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor), dim=-1)
            m = torch.distributions.Categorical(probs)
            A = m.sample()

            board.update_hex(A.item(), self.LABEL)

            log_prob_hex += m.log_prob(A)
            n_items += 1

        return log_prob_digit + (log_prob_hex / n_items)

    def act_greedily(self, board: StrandsBoard):
        """
        Acts greedily without returning anything.
        """
        # choosing a digit
        if board.digits_left_to_place == 0:
            x = board.compute_network_inputs()
            activations = self.get_activations_digits(x)
            mask = board.get_digits_availables()

            mask_tensor = torch.tensor(mask, dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor), dim=-1)
            A = probs.argmax(dim=-1)

            board.update_digit_chosen(A.item())

        # placing tiles on hexes
        while board.digits_left_to_place > 0:
            x = board.compute_network_inputs()
            activations = self.get_activations_hexes(x)
            mask = board.get_hexes_availables()

            mask_tensor = torch.tensor(mask, dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor), dim=-1)
            A = probs.argmax(dim=-1)

            board.update_hex(A.item(), self.LABEL)


class Agent_MLP(Agent):
    def __init__(self, nbHexes: int, nbDigits: int, LABEL_COLOR: int, device: str = "cpu") -> None:
        super().__init__(nbHexes, nbDigits, LABEL_COLOR, device)

        self.mlp = nn.Sequential(
            nn.Linear(nbHexes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.outDigits = nn.Linear(128, nbDigits)
        self.outHexes = nn.Linear(128, nbHexes)

        self.to(device)

    def get_activations_digits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).to(self.device)
        x = self.mlp(x)
        x = F.relu(self.outDigits(x))
        return x.cpu()

    def get_activations_hexes(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).to(self.device)
        x = self.mlp(x)
        x = F.relu(self.outHexes(x))
        return x.cpu()

class Agent_CNN(Agent):
    def __init__(self, nbHexes: int, nbDigits: int, LABEL_COLOR: int, device: str = "cpu") -> None:
        super().__init__(nbHexes, nbDigits, LABEL_COLOR, device)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.outDigits = nn.Linear(nbHexes*64, nbDigits)
        self.outHexes = nn.Linear(nbHexes*64, nbHexes)

        self.to(device)
    
    def get_activations_digits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)
        x = self.cnn(x).flatten(0)
        x = F.relu(self.outDigits(x))
        return x.cpu()

    def get_activations_hexes(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)
        x = self.cnn(x).flatten(0)
        x = F.relu(self.outHexes(x))
        return x.cpu()


class Agent_Random(Agent):
    def __init__(self, nbHexes: int, nbDigits: int, LABEL_COLOR: int, device: str = "cpu") -> None:
        super().__init__(nbHexes, nbDigits, LABEL_COLOR, device)

    def get_activations_digits(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(self.nbDigits)

    def get_activations_hexes(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(self.nbHexes)



def init_agents(board: StrandsBoard, device: str = "cpu", policy: str = "random") -> list[Agent]:
    if policy == "mlp":
        return [Agent_MLP(board.nbHexes, board.LABEL_BLACK+1, board.LABEL_WHITE, device), 
                Agent_MLP(board.nbHexes, board.LABEL_BLACK+1, board.LABEL_BLACK, device)]
    elif policy == "cnn":
        return [Agent_CNN(board.nbHexes, board.LABEL_BLACK+1, board.LABEL_WHITE, device), 
                Agent_CNN(board.nbHexes, board.LABEL_BLACK+1, board.LABEL_BLACK, device)]    
    else:
        return [Agent_Random(board.nbHexes, board.LABEL_BLACK+1, board.LABEL_WHITE, device), 
                Agent_Random(board.nbHexes, board.LABEL_BLACK+1, board.LABEL_BLACK, device)]

