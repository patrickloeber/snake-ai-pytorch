import pygame
import sys
import time
import random
from pygame.locals import *
import numpy as np

from collections import namedtuple
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
            

Point = namedtuple('Point','x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE2 = (0, 100, 255)
BLUE1 = (0, 0, 255)
BLACK = (0,0,0)

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

SPEED = 40

class zz_game_ai:

    def __init__(self, w=640, h=480):        
        self.clock = pygame.time.Clock()

        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Deep Q Snake!')
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)        
        self.snake = [self.head,
                      Point(self.head.x-20, self.head.y), 
                      Point(self.head.x-40, self.head.y)]

        self._place_food()
        self.score = 0
        self.frame_iteration = 0
        
    def _place_food(self):
        s = 20
        x = random.randint(0, (self.w-s)//s)*s
        y = random.randint(0, (self.h-s)//s)*s
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def frame_step(self, action):
        reward, done, score = self.play(action)
        return reward, done, score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
                
        if pt.x > self.w-20 or pt.x < 0 or pt.y > self.h-20 or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
            
        
    # Snake and food
    def play(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))
        
        self.move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            done = True
            reward = -10
            return reward, done, self.score
        
        if self.head == self.food:
            self._place_food()
            self.score += 1
            reward = 10
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, done, self.score
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, Rect(pt.x, pt.y, 20, 20))
            pygame.draw.rect(self.display, BLUE2, Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, Rect(self.food.x, self.food.y, 20, 20))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        # no change
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4 # r -> d -> l -> u
            new_dir = clock_wise[next_idx]
        else: # np.array_equal(action, [0, 0, 1])
            next_idx = (idx - 1) % 4 # r -> u -> l -> d
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
            
        x = self.head.x
        y = self.head.y
        if new_dir == Direction.RIGHT:
            x += 20
        elif new_dir == Direction.LEFT:
            x -= 20
        elif new_dir == Direction.DOWN:
            y += 20
        elif new_dir == Direction.UP:
            y -= 20
            
        self.head = Point(x, y)
            