import numpy as np
from math import copysign,cos,sin,radians
import cv2
import torch
import time

WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (170,170,170)


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
        self.round_idx = 0


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
        d_a =self.get_digits_availables()
        
        if np.sum(d_a)==1:
            for label in range(1,7):
                if d_a[label] and np.sum(self.labels_bitmaps[label])<=label:
                    return True
        

    def update_hex(self,hex,new_label):
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
    

    def compute_score(self, target_label) -> int:
        def bfs(hex, target_label)-> int:
            if visited[hex]:
                return 0
            else:
                visited[hex] = True
                sum = int(self.labels_bitmaps[target_label][hex])

                for neighbour in self.neighbours(hex):
                    sum += bfs(neighbour, target_label)
                return sum
        
        visited = [self.labels_bitmaps[0][hex] for hex in range(self.nbHexes) ]

        max_area = 0
        for hex in range(self.nbHexes):
            if not visited[hex]:
                max_area = max(max_area ,bfs(hex, target_label))

        return max_area
                
    def compute_reward(self) -> int:
        return(copysign(1, self.compute_score(self.LABEL_WHITE)- self.compute_score(self.LABEL_BLACK)) )

    def compute_network_inputs(self)->torch.Tensor:
        return torch.tensor(self.colors,dtype = torch.float32)

    def draw(self,delay_s,scale = 100):
        
        if delay_s == -1:
            return

        def draw_hexagon(x,y, scale,fill_color,label):
            """Draws a hexagon with optional number in its center."""
            angles_deg = [60 * i + 30 for i in range(6)]
            pts = np.array([(int(x +  0.55*scale * cos(radians(angle))),
                        int( y +   0.65*scale * sin(radians(angle)))) for angle in angles_deg])
            if fill_color == 'WHITE':
                fill_color = WHITE
                number_color = BLACK
            elif fill_color == 'BLACK':
                fill_color = BLACK
                number_color = WHITE
            elif fill_color == 'GRAY':
                fill_color = GRAY
                number_color = BLACK
            cv2.fillPoly(img, [pts], fill_color)
            cv2.putText(img, text=f'{label}',org=(int(x-0.1*scale), int(y+0.1*scale)),fontScale=scale/60, color=number_color,thickness=int(scale/20),fontFace=cv2.FONT_HERSHEY_SIMPLEX)


        img = GRAY[0]*np.ones([scale*self.board_size, scale*self.board_size],dtype=np.uint8)
        
        center=self.board_size//2+1
        for hex in range(self.board_size*self.board_size):
            if not self.labels_bitmaps[0][hex]:
                row,col = hex//self.board_size, hex%self.board_size
                y,x = scale*(row+0.5), scale*(col+0.5*(row-center)+1)
                fill_color = ('BLACK' if self.labels_bitmaps[self.LABEL_BLACK][hex]  else ('WHITE' if self.labels_bitmaps[self.LABEL_WHITE][hex] else 'GRAY'))
                draw_hexagon(x,y, scale,fill_color = fill_color,label = self.mapping_hex_to_default_label[hex] )


        cv2.imshow('Display',img) 
        cv2.waitKey(int(1000*delay_s))
        time.sleep(delay_s)



