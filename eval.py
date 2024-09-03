from agent import *
from env import *

def rollout(agents: list[Agent], board: StrandsBoard,display_s:float = -1) -> int:
    """
    Returns +1 for a win for white, -1 for a win for black, 0 for a draw
    """
    board.reset()
    board.draw(display_s)

    with torch.no_grad():
        while not board.check_for_termination():
            i = board.round_idx%2 # 0 for "WHITE to play", 1 for "BLACK to play"
            agents[i].act_greedily(board)
            board.draw(display_s)

    reward = board.compute_reward()

    return(reward)

def evaluate_vs_self(agents: list[Agent], board: StrandsBoard, n_rollouts: int, display_s:float = -1) -> float:

    rewards_agent_WHITE = []
    rewards_agent_BLACK = []

    for i in range(n_rollouts):
        
        reward_agent_WHITE = rollout([agents[0], agents[0]], board, display_s)
        reward_agent_BLACK = rollout([agents[1],agents[1]], board, display_s)

        rewards_agent_WHITE.append(reward_agent_WHITE)
        rewards_agent_BLACK.append(reward_agent_BLACK)


    return(np.mean(rewards_agent_WHITE),np.mean(rewards_agent_BLACK))

def evaluate_vs_random(agents: list[Agent], board: StrandsBoard, n_rollouts: int, display_s:float = -1, device: str = "cpu") -> float:
    
    agents_random = init_random_agents(board)
    
    rewards_agent_WHITE = []
    rewards_agent_BLACK = []

    for i in range(n_rollouts):
        

        reward_agent_WHITE = rollout([agents[0], agents_random[1]], board, display_s)
        reward_agent_BLACK = rollout([agents_random[0],agents[1]], board, display_s)

        rewards_agent_WHITE.append(reward_agent_WHITE)
        rewards_agent_BLACK.append(reward_agent_BLACK)


    return(np.mean(rewards_agent_WHITE),np.mean(rewards_agent_BLACK))
        