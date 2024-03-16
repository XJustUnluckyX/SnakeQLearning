import torch
import random
import numpy as np
from collections import deque
from q_net import QNet
from q_trainer import QNetTrainer
from snake_env import SnakeDirection, Point, BLOCK_SIZE

MAX_MEMORY = 100_000


class QLearner:

    def __init__(self, epsilon, gamma, learning_rate, batch_size, epsilon_decay, epsilon_min):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.num_games = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Inizializzazione del modello e del trainer
        self.deep_model = QNet(11, 256, 3)  # input, hidden, output dims
        self.q_trainer = QNetTrainer(self.deep_model, lr=self.learning_rate, gamma=self.gamma)

        # Inizializzazione della memoria di riproduzione
        self.memory = deque(maxlen=MAX_MEMORY)

    def compute_state(self, status):
        head = status.snake[0]
        pl = Point(head.x - BLOCK_SIZE, head.y)
        pr = Point(head.x + BLOCK_SIZE, head.y)
        pu = Point(head.x, head.y - BLOCK_SIZE)
        pd = Point(head.x, head.y + BLOCK_SIZE)

        # Controllo delle collisioni
        collision_front = status.detect_collision(pr) if status.direction == SnakeDirection.RIGHT else \
            status.detect_collision(pl) if status.direction == SnakeDirection.LEFT else \
                status.detect_collision(pu) if status.direction == SnakeDirection.UP else \
                    status.detect_collision(pd)  # Davanti
        collision_right = status.detect_collision(pr) if status.direction == SnakeDirection.UP else \
            status.detect_collision(pl) if status.direction == SnakeDirection.DOWN else \
                status.detect_collision(pd) if status.direction == SnakeDirection.RIGHT else \
                    status.detect_collision(pu)  # Dx
        collision_left = status.detect_collision(pr) if status.direction == SnakeDirection.DOWN else \
            status.detect_collision(pl) if status.direction == SnakeDirection.UP else \
                status.detect_collision(pd) if status.direction == SnakeDirection.LEFT else \
                    status.detect_collision(pu)  # Sx

        # Direzioni in cui lo snake può muoversi
        sx = 1 if status.direction == SnakeDirection.LEFT else 0
        dx = 1 if status.direction == SnakeDirection.RIGHT else 0
        up = 1 if status.direction == SnakeDirection.UP else 0
        down = 1 if status.direction == SnakeDirection.DOWN else 0

        # Posizione del cibo rispetto alla testa dello snake
        food_left = 1 if status.food.x < status.head.x else 0
        food_right = 1 if status.food.x > status.head.x else 0
        food_up = 1 if status.food.y < status.head.y else 0
        food_down = 1 if status.food.y > status.head.y else 0

        state = [
            collision_front,
            collision_right,
            collision_left,
            sx,
            dx,
            up,
            down,
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=int)

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Pop dell'item a sx se si raggiunge MAX_MEMORY

    def pred_action(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        last_movement = [0, 0, 0]
        if random.random() < self.epsilon:
            movement = random.randint(0, 2)
            last_movement[movement] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.deep_model(state_0)
            movement = torch.argmax(prediction).item()
            last_movement[movement] = 1

        return last_movement

    def learn_with_long_memory(self):
        batch_sample = random.sample(self.memory, k=min(len(self.memory), self.batch_size))  # Limita la dimensione del
        # campione alla dimensione della memoria disponibile
        states, actions, rewards, following_states, done = zip(*batch_sample)
        self.q_trainer.train_net(states, actions, rewards, following_states, done)

    def learn_with_short_memory(self, state, action, reward, following_state, done):
        self.q_trainer.train_net(state, action, reward, following_state, done)
