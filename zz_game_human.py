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

SPEED = 20

class zz_game_human:

    def __init__(self, w=640, h=480):        
        self.clock = pygame.time.Clock()

        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Deep Q Snake!')
        self._reset()

    def _reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)        
        self.snake = [self.head,
                      Point(self.head.x-20, self.head.y), 
                      Point(self.head.x-40, self.head.y)]

        self._place_food()
        self.score = 0
        
    def _place_food(self):
        s = 20
        x = random.randint(0, (self.w-s)//s)*s
        y = random.randint(0, (self.h-s)//s)*s
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
                
        if pt.x > self.w-20 or pt.x < 0 or pt.y > self.h-20 or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
            
        
    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        self.move(self.direction)
        self.snake.insert(0, self.head)

        game_over = False
        
        if self.is_collision():
            game_over = True
            return game_over, self.score
        
        if self.head == self.food:
            self._place_food()
            self.score += 1
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)

        return game_over, self.score
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, Rect(pt.x, pt.y, 20, 20))
            pygame.draw.rect(self.display, BLUE2, Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, Rect(self.food.x, self.food.y, 20, 20))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, direction):            
        self.direction = direction
            
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += 20
        elif direction == Direction.LEFT:
            x -= 20
        elif direction == Direction.DOWN:
            y += 20
        elif direction == Direction.UP:
            y -= 20
            
        self.head = Point(x, y)
        
        
if __name__ == '__main__':
    game = zz_game_human()
    while True:
        game_over, score = game.play_step()
        if game_over == True:
            break
        
    print('final score', score)
    pygame.quit()
        
            