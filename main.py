import time
from env import *
from agent import *
from training import *
from eval import *

def test_speed(device: str = "cpu"):

    board = StrandsBoard()

    agents = init_agents(board, device)
    optimizers = [torch.optim.Adam(agents[i].parameters()) for i in range(2)]
    

    # Measure time for training games
    num_training_games = 10
    start_time = time.time()
    
    for _ in range(num_training_games):
        board.reset()
        reinforce(agents, optimizers, board)
    
    total_time = time.time() - start_time
    average_time_per_game = total_time / num_training_games
    
    print(f"Average time per training game: {average_time_per_game:.6f} seconds with {device} device")


    # Measure time for rollout games
    num_rollout_games = 10
    start_time = time.time()
    
    for _ in range(num_rollout_games):
        board.reset()
        rollout(agents,  board)
    
    total_time = time.time() - start_time
    average_time_per_game = total_time / num_rollout_games
    
    print(f"Average time per rollout game: {average_time_per_game:.6f} seconds with {device} device")

def format_time(time_elapsed) -> str: 
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    seconds = int(time_elapsed % 60)
    formatted_time = f"{hours}h {minutes}m {seconds}s"
    return formatted_time

def main(device: str = "cpu"):

    board = StrandsBoard()
    baseline = 0
    rewards = [0]

    agents = init_agents(board, device)
    optimizers = [torch.optim.Adam(agents[i].parameters()) for i in range(2)]

    n_training_games = 10_000
    freq_eval,size_eval = 50, 100

    st = time.time()
    print("Training starts ...")

    for game in range(1,n_training_games+1):

        baseline = np.mean(rewards)
        reward = reinforce(agents, optimizers, board, baseline)
        rewards.append(reward)

        if game % freq_eval == 0:
            rollout(agents,  board)
            time_elapsed = time.time() - st

            eval_agent_WHITE_vs_BLACK,_  = evaluate_vs_self(agents, board, size_eval, -1)
            win_rate_agent_WHITE_vs_BLACK = 0.5*(eval_agent_WHITE_vs_BLACK +1)

            eval_agent_WHITE_vs_RANDOM,eval_agent_BLACK_vs_RANDOM  = evaluate_vs_random(agents, board, size_eval, -1, device)
            win_rate_agent_WHITE_vs_RANDOM, win_rate_agent_BLACK_vs_RANDOM = 0.5*(eval_agent_WHITE_vs_RANDOM +1), 0.5*(eval_agent_BLACK_vs_RANDOM +1)

            print('\n')
            print(64*'-')
            print("\n")
            print(f"After {game} training episodes and ",format_time(time_elapsed)," elapsed, \n")
            print(f"Average  result  of agent_WHITE against agent_BLACK :  {eval_agent_WHITE_vs_BLACK:.2f}")
            print(f"Average win rate of agent_WHITE against agent_BLACK :  { 100* win_rate_agent_WHITE_vs_BLACK:.0f} % \n")

            print(f"Average  result  of agent_WHITE against Random Agent :  {eval_agent_WHITE_vs_RANDOM:.2f}")
            print(f"Average win rate of agent_WHITE against Random Agent :  { 100* win_rate_agent_WHITE_vs_RANDOM:.0f} % \n")

            print(f"Average  result  of agent_BLACK against Random Agent :  {eval_agent_BLACK_vs_RANDOM:.2f}")
            print(f"Average win rate of agent_BLACK against Random Agent :  { 100* win_rate_agent_BLACK_vs_RANDOM:.0f} %" )

            
if __name__ == "__main__":
    test_speed("cuda")
    test_speed("cpu")
    main("cpu")