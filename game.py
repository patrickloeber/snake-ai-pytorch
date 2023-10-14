import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

RED = (128, 0, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20

class SnakeGameAI:

    def __init__(self, w=640, h=640):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        self.background = pygame.image.load("background.png")
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.base_speed = 10
        self.speed_increase_rate = 1

        self.reset()


    def reset(self):
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
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        
        snake_length = len(self.snake)
        game_speed = self.calculate_speed(snake_length)
        self.clock.tick(game_speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x >= self.w:
            pt = Point(0, pt.y)  # Wrap around to the left side
        elif pt.x < 0:
            pt = Point(self.w - BLOCK_SIZE, pt.y)  # Wrap around to the right side
        if pt.y >= self.h:
            pt = Point(pt.x, 0)  # Wrap around to the top
        elif pt.y < 0:
            pt = Point(pt.x, self.h - BLOCK_SIZE)  # Wrap around to the bottom

        for segment in self.snake[1:]:
            if pt == segment:
                return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)
        self.display.blit(self.background, (0, 0))

        light_green = (0, 255, 0)
        dark_green = (0, 128, 0)

        for i, pt in enumerate(self.snake):
            
            if i % 2 == 0:
                color = light_green
            else:
                color = dark_green

            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()



    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] 

        if (new_dir == Direction.RIGHT and self.direction != Direction.LEFT) or \
        (new_dir == Direction.LEFT and self.direction != Direction.RIGHT) or \
        (new_dir == Direction.UP and self.direction != Direction.DOWN) or \
        (new_dir == Direction.DOWN and self.direction != Direction.UP):
            self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x = (x + BLOCK_SIZE) % self.w
        elif self.direction == Direction.LEFT:
            x = (x - BLOCK_SIZE + self.w) % self.w
        elif self.direction == Direction.DOWN:
            y = (y + BLOCK_SIZE) % self.h
        elif self.direction == Direction.UP:
            y = (y - BLOCK_SIZE + self.h) % self.h

        self.head = Point(x, y)

    def calculate_speed(self, snake_length):
        return self.base_speed + self.speed_increase_rate * snake_length

if __name__ == '__main__':
    game = SnakeGameAI()

    while True:
        action = np.random.choice([0, 1, 2])
        reward, game_over, score = game.play_step(action)

        if game_over:
            break
