import numpy as np
from math import copysign,cos,sin,radians
import cv2
import torch
import time

WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (170,170,170)
BACKGROUND = (70,100,100)

class StrandsBoard:
    def __init__(self, nRings = 6) -> None:
        self.nRings = nRings
        self.board_size = (2*self.nRings-1)
        self.nbHexes = self.board_size * self.board_size
        self.LABEL_WHITE = 7
        self.LABEL_BLACK = 8

        self.labels_bitmaps,self.mapping_hex_to_default_label = self.init_labels_bitmaps()
        self.colors =  np.zeros(self.nbHexes)
    
        self.digit_chosen = 2
        self.digits_left_to_place = 1
        self.round_idx = 1


    def reset(self):
        self.labels_bitmaps,self.mapping_hex_to_default_label = self.init_labels_bitmaps()
        self.colors =  np.zeros(self.nbHexes)
    
        self.digit_chosen = 2
        self.digits_left_to_place = 1
        self.round_idx = 0


    def init_labels_bitmaps(self)->tuple[list[list[bool]], list[int]] : 
        labels_bitmaps = [[False for hex in range(self.nbHexes)] for label in range(self.LABEL_BLACK+1)]
        mapping_hex_to_default_labels = [0 for hex in range(self.nbHexes)] 

        for hex in range(self.nbHexes):

            row = hex// self.board_size - (self.nRings-1) # centered
            col = hex % self.board_size - (self.nRings-1) # centered

            dist_from_center = max(abs(row), abs(col), abs(row + col))
                    
            corners = [(row, col) for row in (-self.nRings+1, self.nRings-1) for col in (-self.nRings+1, self.nRings-1)]
            corners.extend([(-self.nRings+1, 0), (0, -self.nRings+1), (self.nRings-1, 0), (0, self.nRings-1)])
            
            if dist_from_center == 0:
                label = 1

            elif dist_from_center >= self.nRings:
                label = 0

            elif (row, col) in corners:
                label = 6

            elif abs(row) == self.nRings-1 or abs(col) == self.nRings-1 or abs(row + col) == self.nRings-1:
                label = 5

            elif abs(row) == self.nRings - 2 or abs(col) == self.nRings - 2 or abs(row + col) == self.nRings - 2:
                label = 3

            elif abs(row) <= self.nRings - 3 and abs(col) <= self.nRings - 3 and abs(row + col) <= self.nRings - 3:
                label = 2


            labels_bitmaps[label][hex] = True
            mapping_hex_to_default_labels[hex] = label

        return (labels_bitmaps, mapping_hex_to_default_labels)

    def check_for_termination(self) -> bool:
        if self.round_idx == 0:
            return False
        if np.sum(self.get_digits_availables())==0:
            return True
        

    def update_hex(self,hex,new_label):
        assert hex>=0 and hex<self.nbHexes, "impossible to place a tile: hex is out of bounds"
        assert new_label == self.LABEL_WHITE or new_label == self.LABEL_BLACK, "impossible to place a tile: label is not valid"
        assert not self.labels_bitmaps[0][hex], "impossible to place a tile: hex is out of bounds"
        assert self.digit_chosen>0, "impossible to place a tile: digit chosen is not valid"
        assert self.digits_left_to_place>0, "impossible to place a tile: no tiles left for this round"
        assert not self.labels_bitmaps[new_label][hex], "impossible to place a tile: tile already occupied"
        assert self.labels_bitmaps[self.digit_chosen][hex], "impossible to place a tile: the hex doesnt belong to the digit chosen"
        
        self.labels_bitmaps[new_label][hex] = True
        self.labels_bitmaps[self.digit_chosen][hex] = False

        if (new_label == self.LABEL_WHITE):
            self.colors[hex] = 1
        elif (new_label == self.LABEL_BLACK):
            self.colors[hex] = -1
        
        self.digits_left_to_place -= 1
        if self.digits_left_to_place == 0:
            self.round_idx +=1


    def update_digit_chosen(self,new_digit):
        self.digit_chosen = new_digit

        count = 0
        for hex in range(self.nbHexes):
            if self.labels_bitmaps[self.digit_chosen][hex]:
                count+=1
        
        self.digits_left_to_place = min(self.digit_chosen,count)

        assert self.digits_left_to_place>0, "impossible to call this function: the digit is not available because it has no valid free hexes"


    def get_hexes_availables(self) -> list[bool]:
        assert self.digits_left_to_place>0, "impossible to call this function: a digit should already be selected"
        return(self.labels_bitmaps[self.digit_chosen])


    def get_digits_availables(self)-> list[bool]:
        assert self.digits_left_to_place==0, "impossible to call this function: all tiles from previous digit should have been placed"
        is_valid_digit = [False for i in range(len(self.labels_bitmaps))]

        for label in range(1,7):
            for hex in range(self.nbHexes):
                if self.labels_bitmaps[label][hex]:
                    is_valid_digit[label] = True
                    break

        return is_valid_digit


    def neighbours(self,hex):
        x = hex % self.board_size
        y = hex // self.board_size

        neighbours = []
        for direction in [(-1,0),(1,1),(1,0),(-1,-1), (0,-1), (0,1)]:
            x2= max(0, min(x+direction[0],self.board_size-1))
            y2= max(0, min(y+direction[0],self.board_size-1))
            neighbours.append((x2+self.board_size*y2))

        return neighbours
    
    def make_first_random_action(self):
        hexes = self.get_hexes_availables()
        hex = np.random.choice(np.where(hexes)[0])
        self.update_hex(hex, self.LABEL_BLACK)

    def compute_areas(self, target_label) -> int:
        def bfs(hex, target_label)-> int:
            if visited[hex]:
                return 0
            else:
                visited[hex] = True
                if self.labels_bitmaps[target_label][hex]:
                    sum = 1
                    for neighbour in self.neighbours(hex):
                        sum += bfs(neighbour, target_label)
                    return sum
                return 0
        
        visited = [self.labels_bitmaps[0][hex] for hex in range(self.nbHexes) ]

        areas = []
        for hex in range(self.nbHexes):
            if not visited[hex]:
                area = bfs(hex, target_label)
                areas.append(area)

        return areas.sort(reverse=True)
    
    def compute_heuristic_areas(self, target_label) -> int:
        opponent_label = (self.LABEL_BLACK if target_label==self.LABEL_WHITE else self.LABEL_WHITE)        
        def heuristic_bfs(hex, target_label)-> int:
            if visited[hex]:
                return 0
            else:
                visited[hex] = True
                if self.labels_bitmaps[target_label][hex]:
                    sum = 1
                    for neighbour in self.neighbours(hex):
                        sum += heuristic_bfs(neighbour, target_label)
                    return sum
                else:
                    opponent_label = 1 if target_label == 2 else 2
                return 0.5*int(not self.labels_bitmaps[opponent_label][hex])
        
        visited = [self.labels_bitmaps[0][hex] for hex in range(self.nbHexes) ]

        areas = []
        for hex in range(self.nbHexes):
            if not visited[hex]:
                area = heuristic_bfs(hex, target_label)
                areas.append(area)

        return areas.sort(reverse=True)
                      
    def compute_reward(self) -> int:
        areas_white = self.compute_areas(self.LABEL_WHITE)
        areas_black = self.compute_areas(self.LABEL_BLACK)
        for i in range (min(len(areas_white),len(areas_black))):
            if areas_white[i] > areas_black[i]:
                return 1
            elif areas_white[i] < areas_black[i]:
                return -1
        if len(areas_white) > len(areas_black):
            return -1
        elif len(areas_white) < len(areas_black):
            return 1
        return 0
    
    def compute_heuristic_reward(self) -> int:
        reward = self.compute_heuristic_areas(self.LABEL_WHITE) - self.compute_heuristic_areas(self.LABEL_BLACK)
        return(reward)

    def compute_network_inputs(self)->torch.Tensor:
        return torch.tensor(self.colors,dtype = torch.float32)

    def draw(self, display_s = 0, scale = 100):
        """
        Draws the game board with the current state of the game.
        Args:
            display_s (int, optional): The time in seconds to display the board. Defaults to 0. If 0 or -1, the board is not displayed.
            scale (int, optional): The scale of the board. Defaults to 100.
        """

        if display_s == -1 or display_s == 0:
            return

        def draw_hexagon(x,y, scale,fill_color,label):

            """Draws a hexagon with optional number in its center."""
            angles_deg = [60 * i + 30 for i in range(6)]
            pts = np.array([(int(x +  0.55*scale * cos(radians(angle))),
                        int( y +   0.65*scale * sin(radians(angle)))) for angle in angles_deg])
            cv2.fillPoly(img, [pts], fill_color)
            if fill_color == WHITE or fill_color == GRAY:
                cv2.putText(img, text=f'{label}',org=(int(x-0.2*scale), int(y+0.2*scale)),fontScale=scale/60, color=BLACK,thickness=int(scale/20),fontFace=cv2.FONT_HERSHEY_SIMPLEX)

            elif fill_color == BLACK:
                cv2.putText(img, text=f'{label}',org=(int(x-0.2*scale), int(y+0.2*scale)),fontScale=scale/60, color=WHITE,thickness=int(scale/20),fontFace=cv2.FONT_HERSHEY_SIMPLEX)

        img = np.full((scale * self.board_size, scale * self.board_size, 3), BACKGROUND, dtype=np.uint8)
        
        center=self.board_size//2+1
        for hex in range(self.board_size*self.board_size):
                row,col = hex//self.board_size, hex%self.board_size
                y,x = scale*(row+0.5), scale*(col+0.5*(row-center)+1)
                
                if self.labels_bitmaps[self.LABEL_WHITE][hex]:
                    fill_color = WHITE
                elif self.labels_bitmaps[self.LABEL_BLACK][hex]:
                    fill_color = BLACK
                elif  self.labels_bitmaps[0][hex]:
                    fill_color = BACKGROUND
                else:
                    fill_color = GRAY

                draw_hexagon(x,y, scale,fill_color = fill_color,label = self.mapping_hex_to_default_label[hex] )


        cv2.imshow('Display',img) 
        cv2.waitKey(int(1000*display_s))
        time.sleep(display_s)



