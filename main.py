import argparse
import time
from env import *
from agent import *
from training import *
from eval import *
from test import *


def main(nRings: int = 6,
         device: str = "cpu", 
         n_training_games: int = 10_000, 
         freq_eval: int = 1000, 
         learning_rate: float = 0.001, 
         display_rollout: bool = False,
         policy: str = "mlp") -> None:
    """
    The main function that trains two agents using reinforcement learning and evaluates their performance.
    
    Parameters:
    device (str): The device to use for training (default: "cpu").
    n_training_games (int): The number of training games to play (default: 10_000).
    freq_eval (int): The frequency at which to evaluate the agents' performance (default: 100).
    learning_rate (float): The learning rate for the Adam optimizer (default: 0.001).
    display_rollout (bool): Whether to display the rollout of the agents' moves (default: False).
    policy (str): The policy to use for the agents (default: "mlp").
    nRings (int): The number of rings on the board (default: 6).
    """
    
    def format_time(time_elapsed) -> str:
        hours = int(time_elapsed // 3600)
        minutes = int((time_elapsed % 3600) // 60)
        seconds = int(time_elapsed % 60)
        return f"{hours}h {minutes}m {seconds}s"

    board = StrandsBoard(nRings = nRings )
    baseline = 0
    rewards = []
    size_eval = 20

    # Initialize agents with different policies
    agents_NN = init_agents(board, device, policy)
    agents_minimax = init_agents(board, policy="minimax")
    agents_mc = init_agents(board, policy="mc")
    agents_random = init_agents(board, policy="random")

    optimizers = [torch.optim.Adam(agents_NN[i].parameters(), lr=learning_rate) for i in range(2)]

    st = time.time()
    print("Training starts ...")

    for game in range(1, n_training_games + 1):
        baseline = (0 if len(rewards) < 5 else np.mean(rewards))
        reward = reinforce(agents_NN, optimizers, board, baseline)
        rewards.append(reward)

        if game % freq_eval == 0:
            if display_rollout:
                rollout(agents_NN, board, display_s=0.1)
            
            time_elapsed = time.time() - st

            # Evaluation against itself and other agents (minimax, Monte Carlo, random)
            eval_WHITE_vs_BLACK, _   = evaluate(agents_NN, board, size_eval)
            eval_WHITE_vs_minimax, _ = evaluate([agents_NN[0], agents_minimax[1]], board, size_eval)
            _, eval_BLACK_vs_minimax = evaluate([agents_minimax[0], agents_NN[1]], board, size_eval)
            eval_WHITE_vs_mc, _      = evaluate([agents_NN[0], agents_mc[1]], board, size_eval)
            _, eval_BLACK_vs_mc      = evaluate([agents_mc[0], agents_NN[1]], board, size_eval)
            eval_WHITE_vs_random, _  = evaluate([agents_NN[0], agents_random[1]], board, size_eval)
            _, eval_BLACK_vs_random  = evaluate([agents_random[0], agents_NN[1]], board,  size_eval)

            # Calculate win rates
            win_rate_WHITE_vs_BLACK = 0.5 * (eval_WHITE_vs_BLACK + 1)
            win_rate_WHITE_vs_minimax = 0.5 * (eval_WHITE_vs_minimax + 1)
            win_rate_BLACK_vs_minimax = 0.5 * (eval_BLACK_vs_minimax + 1)
            win_rate_WHITE_vs_mc = 0.5 * (eval_WHITE_vs_mc + 1)
            win_rate_BLACK_vs_mc = 0.5 * (eval_BLACK_vs_mc + 1)
            win_rate_WHITE_vs_random = 0.5 * (eval_WHITE_vs_random + 1)
            win_rate_BLACK_vs_random = 0.5 * (eval_BLACK_vs_random + 1)

            print('\n')
            print(64 * '-')
            print("\n")
            print(f"After {game} training episodes and {format_time(time_elapsed)} elapsed, \n")
            
            # NN Agent vs NN Agent
            print(f"Average win rate of agent_WHITE_NN vs agent_BLACK_NN: {100 * win_rate_WHITE_vs_BLACK:.0f}% \n")
            
            # NN Agent_WHITE
            print(f"Average win rate of agent_WHITE_NN vs agent_BLACK_Rd: {100 * win_rate_WHITE_vs_random:.0f}%")
            print(f"Average win rate of agent_WHITE_NN vs agent_BLACK_Mm: {100 * win_rate_WHITE_vs_minimax:.0f}% ")
            print(f"Average win rate of agent_WHITE_NN vs agent_BLACK_MC: {100 * win_rate_WHITE_vs_mc:.0f}% \n")

            # NN Agent_BLACK
            print(f"Average win rate of agent_BLACK_NN vs agent_WHITE_Rd: {100 * win_rate_BLACK_vs_random:.0f}%")
            print(f"Average win rate of agent_BLACK_NN vs agent_WHITE_Mm: {100 * win_rate_BLACK_vs_minimax:.0f}%")
            print(f"Average win rate of agent_BLACK_NN vs agent_WHITE_MC: {100 * win_rate_BLACK_vs_mc:.0f}% \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate agents with reinforcement learning.")
    
    parser.add_argument("--nrings", type=int, default=6, help="Number of rings on the board, including the center hex.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training, either 'cpu' or 'cuda'.")
    parser.add_argument("--n-training-games", type=int, default=10_000, help="Number of training games to play.")
    parser.add_argument("--freq-eval", type=int, default=1000, help="Frequency of evaluations during training.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--display-rollout", action="store_true", help="Display rollout during evaluation if set.")
    parser.add_argument("--test-cuda", action="store_true", help="Test CUDA speed.")
    parser.add_argument("--test-cpu", action="store_true", help="Test CPU speed.")
    parser.add_argument("--policy", type=str, default="mlp", help="Policy to use for the agents, default: 'mlp'.")
    args = parser.parse_args()

    if args.test_cuda:
        test_speed("cuda", policy=args.policy)
    if args.test_cpu:
        test_speed("cpu", policy=args.policy)
    
    main( nRings=args.nrings,
         device=args.device, 
         n_training_games=args.n_training_games, 
         freq_eval=args.freq_eval, 
         learning_rate=args.learning_rate, 
         display_rollout=args.display_rollout, 
         policy=args.policy)
