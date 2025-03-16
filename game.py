import random
from enum import Enum
from collections import namedtuple, deque
from itertools import islice

import pygame
import numpy as np

from utils.constants import BLOCK_SIZE, MAX_IMPROVE_TIME, SPEED, PREVIOUS_MOVES_BUFFER_SIZE

pygame.init()
font = pygame.font.Font('./utils/fonts/arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.recent_moves = deque(maxlen=PREVIOUS_MOVES_BUFFER_SIZE)

        # Init Game
        self.reset()
        
        

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):

        # Update frame interation
        self.frame_iteration += 1

        # Store old head position and dist to food
        old_head = Point(self.head.x, self.head.y)
        old_food_dist = abs(self.food.x - old_head.x) + abs(self.food.y - old_head.y)

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)


        # Keep track of recent moves
        self.recent_moves.append((self.head.x, self.head.y))
        
        # 3. check if game over
        reward = 0
        game_over = False

        # Calculate new distance to food
        new_food_dist = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

        # If collision or no improvement, end game
        if self.is_collision() or self.frame_iteration > MAX_IMPROVE_TIME * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:

            if new_food_dist < old_food_dist:
                reward = 0.1
            else: 
                reward = -0.1

            self.snake.pop()
        
        if self.detect_loop_pattern(self.recent_moves):
            reward -= 0.5
        

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):

        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def detect_loop_pattern(self, moves):
        """Check if the snake has been repeating the same pattern of movements using a deque"""
        # Need at least 8 moves to detect a simple loop
        if len(moves) < 8:
            return False
        
        # Check for repeating patterns of length 2, 3, and 4
        check_patterns = [2, 3, 4]
        
        for pattern_length in check_patterns:
            if len(moves) >= pattern_length * 2:
                # Convert slices to lists for comparison
                last_pattern = list(islice(moves, len(moves) - pattern_length, len(moves)))
                previous_pattern = list(islice(moves, len(moves) - (pattern_length * 2), len(moves) - pattern_length))
                
                if last_pattern == previous_pattern:
                    return True
        
        return False

    def is_path_blocked(self, head, tail):
        """Check if there's a clear path from head to tail"""
        # Simple implementation: check if all spaces adjacent to tail are blocked except one
        # tail = self.snake[-1]
        adjacent_spaces = [
            Point(tail.x + 1, tail.y),
            Point(tail.x - 1, tail.y),
            Point(tail.x, tail.y + 1),
            Point(tail.x, tail.y - 1)
        ]
        
        blocked_spaces = 0
        for point in adjacent_spaces:
            # Check if point is within game bounds
            if point.x < 0 or point.x >= self.w or point.y < 0 or point.y >= self.h:
                blocked_spaces += 1
            # Check if point is occupied by snake body (except head)
            elif point in self.snake[1:]:
                blocked_spaces += 1
        
        # If all adjacent spaces except one are blocked, the path might be at risk
        return blocked_spaces >= 3
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):

        # [straiht, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % len(clock_wise) # Right turn: R -> D -> L -> U
            new_dir = clock_wise[next_idx]
        else: 
            next_idx = (idx - 1) % len(clock_wise) # Left turn: R -> U -> L -> D
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

# if __name__ == '__main__':
#     game = SnakeGameAI()
    
#     # game loop
#     while True:
#         game_over, score = game.play_step()
        
#         if game_over == True:
#             break
        
#     print('Final Score', score)
        
        
#     pygame.quit()