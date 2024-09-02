from agent import *
from env import *

def rollout(agents: list[Agent], board: StrandsBoard,delay_s:float = -1) -> int:
    
    board.draw(delay_s)
    with torch.no_grad():
        while not board.check_for_termination():
            i = board.round_idx%2 # 0 for "WHITE to play", 1 for "BLACK to play"
            act(agents[i],board)
            board.draw(delay_s)

    reward = board.compute_reward()

    return(reward)