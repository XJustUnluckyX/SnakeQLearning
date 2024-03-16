import pygame
import random
from enum import Enum
import numpy as np
from collections import namedtuple

# E' stato utilizzato come riferimento il seguente progetto
# https://github.com/patrickloeber/snake-ai-pytorch/tree/main , poi rifattorizzato

BLOCK_SIZE = 20  # Dimensione della cella

pygame.init()
font = pygame.font.SysFont('arial', 25)

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
SPEED = 5000


class SnakeDirection(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeEnv:

    def __init__(self, w=640, h=480):  # Dimensioni della finestra
        self.direction = None
        self.head = None
        self.snake = None
        self.score = None
        self.food = None
        self.frame_it = None
        self.width = w
        self.height = h
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.respawn_snake()

    def update_render(self):
        self.display.fill(BLACK)

        for point in self.snake:

            if self.head:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN, pygame.Rect(point.x + 4, point.y + 4, 12, 12))

            pygame.draw.rect(self.display, BLUE1, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x + 4, point.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def respawn_snake(self):
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.direction = SnakeDirection.RIGHT
        self.score = 0
        self.food = None
        self.spawn_food()
        self.frame_it = 0

    def spawn_food(self):
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:  # Controlliamo che il cibo non venga generato dove si trova la testa dello snake
            self.spawn_food()

    def detect_collision(self, head=None):
        if head is None:
            head = self.head
        if head in self.snake[1:]:
            return True  # Controllo che lo snake colpisca se stesso
        if head.x > self.width - BLOCK_SIZE or head.x < 0 or head.y > self.height - BLOCK_SIZE or head.y < 0:
            return True  # Controllo che lo snake colpisca un bordo
        return False

    def update_movement(self, action):

        clock_wise = [SnakeDirection.RIGHT, SnakeDirection.DOWN, SnakeDirection.LEFT, SnakeDirection.UP]
        i = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):  # Movimento rettilineo
            new_dir = clock_wise[i]
        elif np.array_equal(action, [0, 1, 0]):  # Movimento a Dx
            next_idx = (i + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0,0,1] Movimento a Sx
            next_idx = (i - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir  # Aggiorno la direzione

        x = self.head.x
        y = self.head.y

        if self.direction == SnakeDirection.RIGHT:  # Spostamento dello snake
            x += BLOCK_SIZE
        elif self.direction == SnakeDirection.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == SnakeDirection.DOWN:
            y += BLOCK_SIZE
        elif self.direction == SnakeDirection.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def step(self, action):
        self.frame_it += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.update_movement(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.detect_collision() or self.frame_it > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()
        self.update_render()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
