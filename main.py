import time
from env import *
from agent import *
from training import *
from eval import *

def main():

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


if __name__ == "__main__":
    main()