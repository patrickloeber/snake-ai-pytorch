import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20

class SnakeGame:
    
    def __init__(self, w=640, h=640):
        self.w = w
        self.h = h
        self.background = pygame.image.load("background.png")

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()

        self.base_speed = 10 
        self.speed_increase_rate = 1
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    if self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    if self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    if self.direction != Direction.UP:
                        self.direction = Direction.DOWN

        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        game_speed = self.calculate_speed()
        self.clock.tick(game_speed)

        game_over = False
        
        if self._is_collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()

        return game_over, self.score

    def _is_collision(self):
   
        if self.head in self.snake[1:]:
            return True

        if self.head.x >= self.w:
            self.head = Point(0, self.head.y) 
        elif self.head.x < 0:
            self.head = Point(self.w - BLOCK_SIZE, self.head.y) 
        elif self.head.y >= self.h:
            self.head = Point(self.head.x, 0)
        elif self.head.y < 0:
            self.head = Point(self.head.x, self.h - BLOCK_SIZE) 

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
    
    def _move(self, direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)
        pygame.display.flip()  # Add this line to update the display


    def calculate_speed(self):
        return self.base_speed + self.speed_increase_rate * len(self.snake)

if __name__ == '__main__':
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()

        snake_length = len(game.snake)
        game_speed = game.calculate_speed()  # Call the method
        game.clock.tick(game_speed)

        if game_over:
            break

    print('Final Score', score)

    pygame.quit()

