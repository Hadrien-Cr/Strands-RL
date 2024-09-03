import time
from env import *
from agent import *
from training import *
from eval import *

def test_speed():

    board = StrandsBoard()

    agents = init_agents(board)
    optimizers = [torch.optim.Adam(agents[i].parameters()) for i in range(2)]
    

    # Measure time for training games
    num_training_games = 100
    start_time = time.time()
    
    for _ in range(num_training_games):
        board.reset()
        train(agents, optimizers, board)
    
    total_time = time.time() - start_time
    average_time_per_game = total_time / num_training_games
    
    print(f"Average time per training game: {average_time_per_game:.6f} seconds")


    # Measure time for rollout games
    num_rollout_games = 100
    start_time = time.time()
    
    for _ in range(num_rollout_games):
        board.reset()
        rollout(agents,  board)
    
    total_time = time.time() - start_time
    average_time_per_game = total_time / num_rollout_games
    
    print(f"Average time per training game: {average_time_per_game:.6f} seconds")

def format_time(time_elapsed) -> str: 
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    seconds = int(time_elapsed % 60)
    formatted_time = f"{hours}h {minutes}m {seconds}s"
    return formatted_time

def main():

    eps = 0.05

    board = StrandsBoard()

    agents = init_agents(board)
    optimizers = [torch.optim.Adam(agents[i].parameters()) for i in range(2)]

    n_training_games = 100_000
    freq_eval,size_eval = 1000, 100

    st = time.time()
    print("Training starts ...")

    for game in range(1,n_training_games+1):

        train(agents, optimizers, board, eps)
        
        if game % freq_eval == 0:

            time_elapsed = time.time() - st

            eval_agent_WHITE,eval_agent_BLACK  = evaluate_vs_random(agents, board, size_eval, -1)
            win_rate_agent_WHITE, win_rate_agent_BLACK = 0.5*(eval_agent_WHITE +1), 0.5*(eval_agent_BLACK +1)

            print('\n',64*'-')
            print("\n")
            print(f"After {game} training episodes and ",format_time(time_elapsed)," elapsed, \n")
            print(f"Average  result  of agent_WHITE against Random Agent :  {eval_agent_WHITE:.2f} \n")
            print(f"Average win rate of agent_WHITE against Random Agent :  { 100* win_rate_agent_WHITE:.0f} % \n")
            print(f"Average  result  of agent_BLACK against Random Agent :  {eval_agent_BLACK:.2f} \n")
            print(f"Average win rate of agent_BLACK against Random Agent :  { 100* win_rate_agent_BLACK:.0f} % \n" )

            
if __name__ == "__main__":
    test_speed()
    main()