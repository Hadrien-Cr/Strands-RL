import argparse
import time
from env import *
from agent import *
from training import *
from eval import *
from test import *


def main(device: str = "cpu", 
         n_training_games: int = 10_000, 
         freq_eval: int = 100, 
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
    
    """
    
    def format_time(time_elapsed) -> str:
        hours = int(time_elapsed // 3600)
        minutes = int((time_elapsed % 3600) // 60)
        seconds = int(time_elapsed % 60)
        return f"{hours}h {minutes}m {seconds}s"

    board = StrandsBoard()
    baseline = 0
    rewards = []

    agents = init_agents(board, device, policy)
        
    optimizers = [torch.optim.Adam(agents[i].parameters(), lr=learning_rate) for i in range(2)]

    st = time.time()
    print("Training starts ...")

    for game in range(1, n_training_games + 1):
        baseline = (0 if len(rewards) < 5 else np.mean(rewards))
        reward = reinforce(agents, optimizers, board, baseline)
        rewards.append(reward)

        if game % freq_eval == 0:
            if display_rollout:
                rollout(agents, board, display_s=0.1)
            
            time_elapsed = time.time() - st

            eval_agent_WHITE_vs_BLACK, _ = evaluate_vs_self(agents, board, freq_eval)
            win_rate_agent_WHITE_vs_BLACK = 0.5 * (eval_agent_WHITE_vs_BLACK + 1)

            eval_agent_WHITE_vs_RANDOM, eval_agent_BLACK_vs_RANDOM = evaluate_vs_random(agents, board, freq_eval, device)
            win_rate_agent_WHITE_vs_RANDOM, win_rate_agent_BLACK_vs_RANDOM = 0.5 * (eval_agent_WHITE_vs_RANDOM + 1), 0.5 * (eval_agent_BLACK_vs_RANDOM + 1)

            print('\n')
            print(64 * '-')
            print("\n")
            print(f"After {game} training episodes and {format_time(time_elapsed)} elapsed, \n")
            print(f"Average result of agent_WHITE against agent_BLACK: {eval_agent_WHITE_vs_BLACK:.2f}")
            print(f"Average win rate of agent_WHITE against agent_BLACK: {100 * win_rate_agent_WHITE_vs_BLACK:.0f}% \n")
            print(f"Average result of agent_WHITE against Random Agent: {eval_agent_WHITE_vs_RANDOM:.2f}")
            print(f"Average win rate of agent_WHITE against Random Agent: {100 * win_rate_agent_WHITE_vs_RANDOM:.0f}% \n")
            print(f"Average result of agent_BLACK against Random Agent: {eval_agent_BLACK_vs_RANDOM:.2f}")
            print(f"Average win rate of agent_BLACK against Random Agent: {100 * win_rate_agent_BLACK_vs_RANDOM:.0f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate agents with reinforcement learning.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training, either 'cpu' or 'cuda'.")
    parser.add_argument("--n-training-games", type=int, default=10_000, help="Number of training games to play.")
    parser.add_argument("--freq-eval", type=int, default=100, help="Frequency of evaluations during training.")
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
    
    main(device=args.device, 
         n_training_games=args.n_training_games, 
         freq_eval=args.freq_eval, 
         learning_rate=args.learning_rate, 
         display_rollout=args.display_rollout, 
         policy=args.policy)
