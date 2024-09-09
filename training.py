from agent import *
from env import *
import torch.nn.functional as F

def reinforce(agents: list[Agent_NN], optimizers, board: StrandsBoard, baseline: float = 0.0) -> int:
    """
    Apply the training logic on one game 
    REINFORCE
    """

    board.reset()
    board.make_first_random_action()
    
    saved_log_probs_WHITE = []
    saved_log_probs_BLACK = []  
    
    while not board.check_for_termination():
        
        i = board.round_idx % 2  # 0 for "WHITE to play", 1 for "BLACK to play"

        log_prob = agents[i].act_reinforce(board)

        if i == 0:
            saved_log_probs_WHITE.append(log_prob)
        else:
            saved_log_probs_BLACK.append(log_prob)

    # Handle final step
    reward = board.compute_reward()

    policy_loss_WHITE = []
    for log_prob in saved_log_probs_WHITE:
        policy_loss_WHITE.append(-log_prob * (reward - baseline))
    policy_loss_WHITE = torch.cat(policy_loss_WHITE).sum()

    policy_loss_BLACK = []
    for log_prob in saved_log_probs_BLACK:
        policy_loss_BLACK.append(-log_prob * (-reward - baseline))
    policy_loss_BLACK = torch.cat(policy_loss_BLACK).sum()

    optimizers[0].zero_grad()
    policy_loss_WHITE.backward()
    optimizers[0].step()

    optimizers[1].zero_grad()
    policy_loss_BLACK.backward()
    optimizers[1].step()


    return(reward, policy_loss_WHITE.item(), policy_loss_BLACK.item())