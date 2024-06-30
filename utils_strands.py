
import math
import cv2
import numpy as np

gray_value = 170
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (gray_value,gray_value,gray_value)

def inverse_flatten(idx,size):
    return(idx%size,idx//size)

def flatten(row,col,size):
    return row*size+col

def get_hex_label(row, col, max_dist):
    dist_from_center = max(abs(row), abs(col), abs(row + col))
    if dist_from_center == 0:
        return 1
    label = 5
        
    corners = [(row, col) for row in (-max_dist, max_dist) for col in (-max_dist, max_dist)]
    corners.extend([(-max_dist, 0), (0, -max_dist), (max_dist, 0), (0, max_dist)])
    if (row, col) in corners:
        label = 6
    elif abs(row) == max_dist or abs(col) == max_dist or abs(row + col) == max_dist:
        label = 5
    elif abs(row) == max_dist - 1 or abs(col) == max_dist - 1 or abs(row + col) == max_dist - 1:
        label = 3
    elif abs(row) <= max_dist - 2 and abs(col) <= max_dist - 2 and abs(row + col) <= max_dist - 2:
        label = 2
    return label

def mapping_hex_to_label(size):
    labels = []
    for idx in range(size * size):
        (row, col) = inverse_flatten(idx, size)
        if row+col<size//2 or row+col>=3*size//2: labels.append(0)
        else:labels.append(get_hex_label(row-size//2, col-size//2, size//2))
    return labels

def compute_next_player(buffer,count,player):
    """Check if the turn has ended"""
    if count==0:
        return(1-player)
    else:
        _,label = buffer[-1]
        if buffer[-label::]==[(player,label)]*label:
             return(1-player)
        else:
            return(player)

def draw_all(img,scale,size,mapping_hex_label,board):

    def draw_hexagon( x, y, scale,fill_color,label):
        """Draws a hexagon with optional number in its center."""
        angles_deg = [60 * i + 30 for i in range(6)]
        pts = np.array([(int(x +  0.55*scale * math.cos(math.radians(angle))),
                       int( y +   0.65*scale * math.sin(math.radians(angle)))) for angle in angles_deg])
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

    center=size//2+1
    for idx in range(size*size):
        if mapping_hex_label[idx]!= 0:
            row,col = idx//size, idx%size
            y,x = scale*(row+0.5), scale*(col+0.5*(row-center)+1)
            fill_color = ('GRAY' if board[idx] == 2 else ('WHITE' if board[idx] == 1 else 'BLACK'))
            draw_hexagon(x,y, scale,fill_color = fill_color,label = mapping_hex_label[idx] )
            
    return(img)
    
