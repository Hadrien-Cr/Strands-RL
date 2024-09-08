import torch
import torch.nn as nn
import torch.nn.functional as F
from env import *
import itertools


class Agent:
    def __init__(self, board: StrandsBoard, LABEL_COLOR: int) -> None:
        self.board = board
        self.LABEL = LABEL_COLOR
        self.nbHexes = board.nbHexes
        self.nbDigits = board.nbDigits
        self.board_size = board.board_size

    def act_greedily(self, board: StrandsBoard):
        """
        Acts greedily without returning anything.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class Agent_NN(Agent, nn.Module):
    def __init__(self, board: StrandsBoard, LABEL_COLOR: int, device: str = "cpu") -> None:
        Agent.__init__(self, board, LABEL_COLOR)
        nn.Module.__init__(self)
        self.device = device

    def get_activations_digits(self, state) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_activations_hexes(self, state) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def act_reinforce(self, board: StrandsBoard) -> torch.Tensor:
        """
        Acts and returns the (differentiable) log probability of the picked action.
        """
        log_prob_digit, log_prob_hex, n_items = torch.tensor([0.]), torch.tensor([0.]), 0

        # choosing a digit
        if board.digits_left_to_place == 0:
            state = board.compute_board_state()
            activations = self.get_activations_digits(state)
            mask = state['mask']

            mask_tensor = torch.tensor(mask, dtype=torch.float)
            probs = F.softmax(activations + torch.log(mask_tensor), dim=-1)
            m = torch.distributions.Categorical(probs)
            A = m.sample()

            board.update_digit_chosen(A.item())
            log_prob_digit = m.log_prob(A)

        # placing tiles on hexes
        while board.digits_left_to_place > 0:
            state = board.compute_board_state()
            activations = self.get_activations_hexes(state)
            mask = state['mask']

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
            state = board.compute_board_state()
            activations = self.get_activations_digits(state)

            mask = state['mask']
            mask_tensor = torch.tensor(mask, dtype=torch.float)

            probs = F.softmax(activations + torch.log(mask_tensor), dim=-1)
            A = probs.argmax(dim=-1)

            board.update_digit_chosen(A.item())

        # placing tiles on hexes
        while board.digits_left_to_place > 0:
            state = board.compute_board_state()
            activations = self.get_activations_hexes(state)

            mask = state['mask']
            mask_tensor = torch.tensor(mask, dtype=torch.float)

            probs = F.softmax(activations + torch.log(mask_tensor), dim=-1)
            A = probs.argmax(dim=-1)

            board.update_hex(A.item(), self.LABEL)


class Agent_MLP(Agent_NN):
    def __init__(self, board: StrandsBoard, LABEL_COLOR: int, device: str = "cpu") -> None:
        super().__init__(board, LABEL_COLOR, device)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.nbHexes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.outDigits = nn.Linear(128, self.nbDigits)
        self.outHexes = nn.Linear(128, self.nbHexes)

        self.to(device)

    def get_activations_digits(self, state) -> torch.Tensor:
        x = torch.tensor(state["colors"], dtype=torch.float).view(self.board_size, self.board_size).unsqueeze(0).to(self.device)
        x = self.mlp(x)
        x = F.relu(self.outDigits(x))
        return x.cpu()

    def get_activations_hexes(self, state) -> torch.Tensor:
        x = torch.tensor(state["colors"], dtype=torch.float).view(self.board_size, self.board_size).unsqueeze(0).to(self.device)
        x = self.mlp(x)
        x = F.relu(self.outHexes(x))
        return x.cpu()


class Agent_CNN(Agent_NN):
    def __init__(self, board: StrandsBoard, LABEL_COLOR: int, device: str = "cpu") -> None:
        super().__init__(board, LABEL_COLOR, device)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.outDigits = nn.Linear(self.nbHexes * 64, self.nbDigits)
        self.outHexes = nn.Linear(self.nbHexes * 64, self.nbHexes)

        self.to(device)
    
    def get_activations_digits(self, state) -> torch.Tensor:
        x = torch.tensor(state["colors"], dtype=torch.float).view(self.board_size, self.board_size).unsqueeze(0).to(self.device)
        x = self.cnn(x).flatten(0)
        x = F.relu(self.outDigits(x))
        return x.cpu()

    def get_activations_hexes(self, state) -> torch.Tensor:
        x = torch.tensor(state["colors"], dtype=torch.float).view(self.board_size, self.board_size).unsqueeze(0).to(self.device)
        x = self.cnn(x).flatten(0)
        x = F.relu(self.outHexes(x))
        return x.cpu()


class Agent_Random(Agent):
    def __init__(self, board: StrandsBoard, LABEL_COLOR: int) -> None:
        super().__init__(board, LABEL_COLOR)

    def act_greedily(self, board: StrandsBoard):
        # Random action for digits
        if board.digits_left_to_place == 0:
            state = board.compute_board_state()
            mask = state['mask']
            A  = np.random.choice(np.where(mask)[0])
            board.update_digit_chosen(A)

        # Random action for hexes
        while board.digits_left_to_place > 0:
            state = board.compute_board_state()
            mask = state['mask']
            A  = np.random.choice(np.where(mask)[0])
            board.update_hex(A, self.LABEL)


class Agent_1StepMinimax(Agent):
    def __init__(self, board: StrandsBoard, LABEL_COLOR: int, budget: float = 0.1) -> None:
        super().__init__(board, LABEL_COLOR)
        self.budget = budget

    def exhaustive_1Stepsearch(self, board: StrandsBoard):

        assert(board.digits_left_to_place ==0)
        scores = [[(i,None,-1000)] for i in range(board.nbDigits)]
        root_state = board.compute_board_state()
        digits_av = root_state['mask']

        start_time = time.time()
        for digit in range(board.nbDigits):
            
            if digits_av[digit]:
                board.update_digit_chosen(digit)
                intermediate_state = board.compute_board_state()
                hexes_av = intermediate_state['mask']
                hexes = np.where(hexes_av)[0]
                combinations = itertools.combinations(hexes, board.digits_left_to_place)
                for combination in combinations:
                    if (self.budget < time.time() - start_time) :
                        break
                    board.restore_board_state(state=intermediate_state)
                    for hex in combination:
                        board.update_hex(hex, self.LABEL)

                    score = (-1 if self.LABEL == board.LABEL_WHITE else 1) * board.compute_heuristic_reward(self.LABEL)
                    scores[digit].append((digit,combination,score))
                board.restore_board_state(state=root_state)

        scores_grouped = [max(scores[digit], key = lambda x: x[2]) for digit in range(board.nbDigits)]
        return max(scores_grouped, key = lambda x: x[2])

    def act_greedily(self, board: StrandsBoard):
        assert (board.digits_left_to_place == 0)
        digit,combination,score = self.exhaustive_1Stepsearch(board)
        board.update_digit_chosen(digit)
        for hex in combination:
            board.update_hex(hex, self.LABEL)


class Agent_1StepMC(Agent):
    def __init__(self, board: StrandsBoard, LABEL_COLOR: int, budget: float = 0.1) -> None:
        super().__init__(board, LABEL_COLOR)
        self.budget = budget
        self.default_policy_agents = [Agent_Random(board, LABEL_COLOR), Agent_Random(board, LABEL_COLOR)]
    
    def mc_rollout(self,board: StrandsBoard) -> int:
        with torch.no_grad():
            while not board.check_for_termination():
                i = board.round_idx%2 # 0 for "WHITE to play", 1 for "BLACK to play"
                self.default_policy_agents[i].act_greedily(board)

        reward = board.compute_reward()
        return(reward)
    
    def exhaustive_1Stepsearch(self, board: StrandsBoard):

        assert(board.digits_left_to_place ==0)

        scores = [[(i,None,-1000,0)] for i in range(board.nbDigits)]
        root_state = board.compute_board_state()
        digits_av = root_state['mask']
        
        start_time = time.time()

        # initialize scores
        for digit in range(board.nbDigits):
            if digits_av[digit]:
                board.update_digit_chosen(digit)
                intermediate_state = board.compute_board_state()
                hexes_av = intermediate_state['mask']
                hexes = np.where(hexes_av)[0]
                combinations = itertools.combinations(hexes, board.digits_left_to_place)
                for combination in combinations:
                    if (self.budget < time.time() - start_time) :
                        break    
                    board.restore_board_state(state=intermediate_state)
                    for hex in combination:
                        board.update_hex(hex, self.LABEL)
                    score = (-1 if self.LABEL == board.LABEL_WHITE else 1) * self.mc_rollout(board)
                    scores[digit].append((digit,combination,score,1))

                board.restore_board_state(state=root_state)
        
        while self.budget > (time.time() - start_time):
            for digit in range(board.nbDigits):
                if digits_av[digit]:
                    board.update_digit_chosen(digit)
                    intermediate_state = board.compute_board_state()
                    for idx in range(len(scores[digit])):
                        digit,combination,score,visits =scores[digit][idx]
                        if (self.budget < (time.time() - start_time)):
                            break
                        
                        elif (combination is not None):
                            board.restore_board_state(state=intermediate_state)
                            for hex in combination:
                                board.update_hex(hex, self.LABEL)
                            new_score = (-1 if self.LABEL == board.LABEL_WHITE else 1) * self.mc_rollout(board)
                            scores[digit][idx]  = ((digit,combination,new_score+score,visits+1))
                        board.restore_board_state(state=root_state)
        scores_grouped = [max(scores[digit], key = lambda x: x[2]/max(x[3],1)) for digit in range(board.nbDigits)]
        return max(scores_grouped, key = lambda x: x[2]/max(x[3],1))

    def act_greedily(self, board: StrandsBoard):
        assert (board.digits_left_to_place == 0)
        digit,combination,score,visits = self.exhaustive_1Stepsearch(board)
        board.update_digit_chosen(digit)
        for hex in combination:
            board.update_hex(hex, self.LABEL)

def init_agents(board: StrandsBoard, device: str = "cpu", policy: str = "random", **kwargs) -> list[Agent]:
    if policy == "mlp":
        return [Agent_MLP(board, board.LABEL_WHITE, device), 
                Agent_MLP(board, board.LABEL_BLACK, device)]
    elif policy == "cnn":
        return [Agent_CNN(board, board.LABEL_WHITE, device), 
                Agent_CNN(board, board.LABEL_BLACK, device)]    
    elif policy == "minimax":
        return [Agent_1StepMinimax(board, board.LABEL_WHITE), 
                Agent_1StepMinimax(board, board.LABEL_BLACK)]
    elif policy == "mc":
        return [Agent_1StepMC(board, board.LABEL_WHITE), 
                Agent_1StepMC(board, board.LABEL_BLACK)]
    else:
        return [Agent_Random(board, board.LABEL_WHITE), 
                Agent_Random(board, board.LABEL_BLACK)]


if __name__ == "__main__":
    for nRings in range(4,8):
        start_time = time.time()
        print(f"nRings = {nRings}")
        board = StrandsBoard(nRings)
        board.make_first_random_action()
        agents = init_agents(board, policy = "minimax")
        agents[0].act_greedily(board)
        print(f"Test of Mmax passed in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        board.reset()
        board.make_first_random_action()
        agents = init_agents(board, policy = "mc")
        agents[0].act_greedily(board)
        print(f"Test of MC passed in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        board.reset()
        board.make_first_random_action()
        agents = init_agents(board, policy = "cnn")
        agents[0].act_greedily(board)
        print(f"Test of CNN passed in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        board.reset()
        board.make_first_random_action()
        agents = init_agents(board, policy = "mlp")
        agents[0].act_greedily(board)
        print(f"Test of MLP passed in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        board.reset()
        board.make_first_random_action()
        agents = init_agents(board, policy = "random")
        agents[0].act_greedily(board)
        print(f"Test of Random passed in {time.time() - start_time:.2f} seconds")

