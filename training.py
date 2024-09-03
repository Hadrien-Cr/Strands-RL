from agent import *
from env import *
import torch.nn.functional as F

def train(agents: list[Agent], optimizers, board: StrandsBoard, eps: float = 0.1):
    """
    Apply the training logic on one game 
    """

    board.reset()
    prevQValues = [torch.tensor(0.), torch.tensor(0.)]
    currentQValues = [torch.tensor(0.), torch.tensor(0.)]
    
    while True:
    
        i = board.round_idx % 2  # 0 for "WHITE to play", 1 for "BLACK to play"

        optimizers[i].zero_grad()
        prevQValues[i] = currentQValues[i].clone().detach()
        currentQValues[i] = act(agents[i], board, eps=eps)
        if board.check_for_termination():
            break
        loss = (currentQValues[i] - prevQValues[i]).pow(2).sqrt()
        loss.backward()
        optimizers[i].step()

    # Perform the last action (only one legal move left)
    i = board.round_idx % 2
    prevQValues[i] = currentQValues[i].clone().detach()
    currentQValues[i] = act(agents[i], board)

    # Handle final step
    reward = board.compute_reward()

    for i in range(2):
        if i == 0:
            delta = currentQValues[i] - torch.tensor(reward)
        else:
            delta = currentQValues[i] - torch.tensor(-reward)

        loss = delta.pow(2).sqrt()
        loss.backward()
        optimizers[i].step()

