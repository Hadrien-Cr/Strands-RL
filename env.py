import gymnasium
from gymnasium.spaces import Tuple, Discrete,MultiDiscrete
import numpy as np
from collections import Counter
from collections import deque
from utils_strands import *
import time

class Strands_GymEnv(gymnasium.Env):
    """
    The `Strands_GymEnv` class is a subclass of `gymnasium.Env` and represents an environment for playing the Strands game.

    Attributes:
        single_action_space (Discrete): The action space of the environment, which is a discrete space with `size * size` possible actions.
        single_observation_space (MultiDiscrete): The observation space of the environment, which is a multi-discrete space with 4 * `size * size` possible values.
            This represents the concatenation of 4 bitmaps: 
                - the bitmap that represent the black hexes,
                - the bitmap that represent the white hexes, 
                - the bitmap that represent the empty hexes, 
                - and the bitmap that represent "the mask", the spots where the player can place an hex.
        board_size (int): The size of the game board.

    Methods:
        __init__(size): Initializes the `Strands_GymEnv` object with the given `size` of the game board.
        close(): Destroys all windows created by the environment.
        reset_mask(): Resets the current mask of the environment.
        reset(): Resets the environment to its initial state.
        step(hex): if hex is an integer, performs the action 'placing the hex on the ith spot of the board` and update the current state of the environment. 
            returns self.obs, reward, done, info,
            done indicates if the player has played his last move
            et info['end_of_game'] indicates if the game has ended
        is_legal(action): Checks if the given `action` is legal in the current state of the environment.
        render(mode='human'): Renders the environment in the specified `mode`. If `mode` is 'human', it displays the environment using OpenCV. If `mode` is 'rgb_array', it returns the rendered environment as an RGB array.
        draw_board(scale=50): Generate the image of the game board on an image with the specified `scale`.
        init_remaining_hexes(): Initializes the remaining hexes of the environment.
    """
    def __init__(self,size):
        super().__init__()
        # Define action and observation space
        self.single_action_space = Discrete(size * size)
        self.single_observation_space = MultiDiscrete([2]*(4*size*size))
        self.board_size = size
    
    def calculate_connected_areas(self, owner, fill = None):
        """Calculates the largest connected area of hexes of the specified owner."""
        def dfs(idx):
            """Depth-First Search to count the size of the connected area."""
            if (idx<0 or idx>=len(self.board) or idx in visited) or \
                ((self.board[idx] != owner) and (not (fill==owner and self.board[idx] == 2 and self.bit_map0[idx]==0))):

                visited.add(idx)
                return 0
            
            visited.add(idx)
            count = 1
            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, -1)]
            for dr, dc in directions:
                next_idx = idx + dr + self.board_size*dc
                count += dfs(next_idx)
            return count
        
        visited = set()
        max_area = 0
        
        for idx in range(len(self.board)):
            area = dfs(idx)
            max_area = max(max_area, area)
        
        return max_area
    
    def heuristic_calculate_connected_areas(self,owner):    
        """Optimistic heuristic to calculate the largest connected area of hexes of the specified owner."""
        threshold=0.2
        decay_factor = 0.7
        opponent=1-owner
        empty_hex_value = 0.5
        penalty_factor = 1

        def bfs(idx):
            """Breadth-First Search to count the size of the connected area and propagate through the empty hexes"""
            queue = deque([(idx, 1)])  # (row, col, current_decay)
            visited.add(idx)
            area=0
            while queue:
                idx, decay = queue.popleft()
                # Calculate area based on the decay factor
                if self.board[idx] == owner:
                    area += decay
                elif self.board[idx] == 2 and self.bit_map0[next_idx]==0:   # Uncaptured hex
                    area += decay * empty_hex_value
                # Process neighbors
                directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, -1)]
                for dr, dc in directions:
                    next_idx = idx + dr + self.board_size*dc
                    if next_idx>=0 and next_idx<len(self.board) and self.bit_map0[next_idx]==0: 
                        if next_idx not in visited and self.board[next_idx] != opponent:
                            visited.add(next_idx)
                            new_decay = 1 if self.board[next_idx] == owner else decay * decay_factor #reset the decay to 0 if the tile is the right color
                            if new_decay>threshold:
                                queue.append((next_idx, new_decay))
            return area
        
        # Initialization
        visited = set()
        max_expected_area,penalty = 0,0

        # Determine visit order based on distance from the center
        center = self.board_size//2 +1
        visit_order = sorted(range(len(self.board)), key=lambda x: -abs(x//self.board_size - center) -abs(x%self.board_size -center))
        
        # Perform BFS from each hex of the right color
        for idx in visit_order:
            if idx not in visited and self.board[idx] == owner:
                area = bfs(idx)
                if area > max_expected_area:
                    penalty+=max_expected_area
                    max_expected_area = area
                else:
                    penalty+=area
        
        return max_expected_area-penalty_factor*penalty

    def init_remaining_hexes(self):
        """
        Initializes the remaining hexes with default values based on mapping_hex_to_label.
        """
        l=[0]*7
        for i in range(7):
            l[i] = (Counter(self.mapping_hex_to_label)[i]   if i in set(self.mapping_hex_to_label) else 0) 
        return l
    
    def create_bit_map(self):
        """
        Create a bit map for each label in the mapping_hex_to_label array.

        Returns:
            bit_map0 (np.ndarray): A bit map where each index corresponds to a label in the mapping_hex_to_label array.
            bit_map1 (np.ndarray): A bit map where each index corresponds to a label in the mapping_hex_to_label array.
            bit_map2 (np.ndarray): A bit map where each index corresponds to a label in the mapping_hex_to_label array.
            bit_map3 (np.ndarray): A bit map where each index corresponds to a label in the mapping_hex_to_label array.
            bit_map5 (np.ndarray): A bit map where each index corresponds to a label in the mapping_hex_to_label array.
            bit_map6 (np.ndarray): A bit map where each index corresponds to a label in the mapping_hex_to_label array.
        """

        bit_map0 = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        bit_map1 = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        bit_map2 = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        bit_map3 = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        bit_map5 = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        bit_map5 = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        bit_map6 = np.zeros((self.board_size*self.board_size), dtype=np.int32)

        for i in range(self.board_size*self.board_size):
            if self.mapping_hex_to_label[i] == 0:
                bit_map0[i] = 1
            if self.mapping_hex_to_label[i] == 1:
                bit_map1[i] = 1
            if self.mapping_hex_to_label[i] == 2:
                bit_map2[i] = 1
            if self.mapping_hex_to_label[i] == 3:
                bit_map3[i] = 1
            if self.mapping_hex_to_label[i] == 5:
                bit_map5[i] = 1
            if self.mapping_hex_to_label[i] == 6:
                bit_map6[i] = 1

        return bit_map0,bit_map1,bit_map2,bit_map3,bit_map5,bit_map6


    def reset_mask(self):
        self.current_mask = np.bitwise_and(np.bitwise_not(self.bit_map0),self.bitmap_empty)
    
    def render_bitmap(self,bitmap):
        print('-'*20)
        for row in range(self.board_size):
            print(bitmap[self.board_size*row:self.board_size*(row+1)])

    def reset(self,seed=None, options=None):
        """
        Reset the game state to its initial state.

        Args:
            seed (int, optional): The seed to use for random number generation. Defaults to None.
            options (dict, optional): Additional options for the reset. Defaults to None.

        Returns:
            tuple obs, reward, done, info,
            done indicates if the player has played his last move
            et info['end_of_game'] indicates if the game has ended.

        """
        # Reset board to initial state
        self.board = 2*np.ones((self.board_size*self.board_size), dtype=np.int32)
        self.mapping_hex_to_label = mapping_hex_to_label(self.board_size)
        self.remaining_hexes = self.init_remaining_hexes()
        self.img = None

        # Initialize the bitmaps
        self.bitmap_empty = np.ones((self.board_size*self.board_size), dtype=np.int32)
        self.bitmap_b = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        self.bitmap_w = np.zeros((self.board_size*self.board_size), dtype=np.int32)
        self.bit_map0,self.bit_map1,self.bit_map2,self.bit_map3,self.bit_map5,self.bit_map6 = self.create_bit_map()
        self._dict_bitmaps = {0:self.bit_map0,1:self.bit_map1,2:self.bit_map2,3:self.bit_map3,5:self.bit_map5,6:self.bit_map6}

        # make the first move
        self.remaining_hexes[0] = 0
        self.mapping_hex_to_label[0] = 2
        self.bitmap_b[0] = 1
        self.env_buffer = [(0,2)]
        self.reset_mask()
        self.current_mask = np.bitwise_and(self.current_mask,self.bit_map2)
        self.prev_player=0
        self.player_to_play=0
        self.max_rounds = self.compute_max_rounds(self.remaining_hexes)
        self.current_round = 0

        self.obs = np.concatenate((self.bitmap_empty,self.bitmap_b,self.bitmap_w,self.current_mask),axis=0)
        
        return self.obs,{'end of game':False}
    
    def step(self, hex):
        assert self.is_legal(hex), "Illegal move"
        ################ UPDATE THE STATE ###################
        label = self.mapping_hex_to_label[hex] 
        self.board[hex] = self.player_to_play # update the board
        self.remaining_hexes[label]-=1
        self.env_buffer.append((self.player_to_play,label))
    
        self.bitmap_empty[hex] = 0 # update the bitmaps
        if self.player_to_play == 0: # update the bitmaps
            self.bitmap_b[hex] = 1 # update the bitmaps
        else:
            self.bitmap_w[hex] = 1 # update the bitmaps

        ################ CHANGE WHO IS NEXT ####################
        next_player = compute_next_player(self.env_buffer,self.remaining_hexes[label],self.player_to_play)
        self.prev_player = self.player_to_play
        self.player_to_play = next_player

        # if the player changed, reset the mask
        if self.player_to_play != self.prev_player:
            self.current_round+=1
            self.reset_mask()
             
        # else, update the mask to account for the label
        else:
            self.current_mask[hex]=0
            self.current_mask = np.bitwise_and(self.current_mask,self._dict_bitmaps[label])

        ################ OUTPUT THE NEXT OBS ###################
        done =  (self.max_rounds-self.current_round<=1)
        self.obs = np.concatenate((self.bitmap_empty,self.bitmap_b,self.bitmap_w,self.current_mask),axis=0)


        if (self.current_round == self.max_rounds) and self.player_to_play != self.prev_player:  # the last action of the before last round
            done = True
            reward = self.calculate_connected_areas(owner=0, fill = self.player_to_play)-self.calculate_connected_areas(owner=1, fill = self.player_to_play)
            return self.obs, reward, done, {'end of game':False}
        
        elif (self.current_round == self.max_rounds+1) and np.sum(self.remaining_hexes)==0: # the last action of the game
            done = True
            reward = self.calculate_connected_areas(owner=0)-self.calculate_connected_areas(owner=1)
            return self.obs, reward, done, {'end of game':True}
        
        else:   
            done = False
            reward = 0
            #reward = 0.01*(self.heuristic_calculate_connected_areas(owner=0)-self.heuristic_calculate_connected_areas(owner=1))
            return self.obs, reward, done, {'end of game':False}
        
    def draw_board(self,scale = 50):
        if self.img is None:
            self.img = gray_value*np.ones([scale*self.board_size, scale*self.board_size],dtype=np.uint8)
        self.img = draw_all(img = self.img,scale=scale,size=self.board_size,mapping_hex_label=self.mapping_hex_to_label,board=self.board)
    
    def render(self, mode='human'):
        # Optional: Render the environment
        if mode == 'human':
            self.draw_board()
            cv2.imshow('Render',self.img)
            cv2.waitKey(10)
            time.sleep(0.2)
        elif mode == 'rgb_array':
            self.draw_board()
            return(self.img)
        
        else:
            super().render(mode=mode)

    def close(self):
        cv2.destroyAllWindows()

    def is_legal(self,action):
        return(self.current_mask[action] == 1)
    
    def compute_max_rounds(self,remaining_hexes):
        max_rounds = 1
        for i in range(1,len(remaining_hexes)):
            if i == 2:
                max_rounds += (remaining_hexes[i]-1)//i +(1 if remaining_hexes[i]%i!=0 else 0)
            else:
                max_rounds += (remaining_hexes[i])//i +(1 if remaining_hexes[i]%i!=0 else 0)
        return(max_rounds)
    
import random
if __name__ == '__main__':
    size = 11
    env = Strands_GymEnv(size)
    for _ in range(10):
        obs,info = env.reset()
        env.render()
        while not info['end of game']:
            action = 0
            while not env.is_legal(action):
                action = random.randint(0,size*size-1)
            obs, reward, done, info = env.step(action)
            env.render()
            
        env.close()
