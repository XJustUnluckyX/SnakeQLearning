import torch
import random
import numpy as np
from collections import deque  # Per memorizzare
from deep_model import Linear_QNet, QTrainer
from env import Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # learning rate


# TODO implementare una GridSearch per gli iper-parametri

class QLearner:

    def __init__(self):
        self.epsilon = 0
        self.gamma = 0.9
        self.n_games = 0

        self.deep_model = Linear_QNet(11, 256, 3)  # input, hidden, output dim
        self.q_trainer = QTrainer(self.deep_model, lr=LR, gamma=self.gamma)
        self.memory = deque(maxlen=MAX_MEMORY)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_sx = game.direction == Direction.LEFT
        dir_dx = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        # Controllo delle collisioni
        state = [
            (dir_dx and game.check_collision(point_r)) or  # Davanti
            (dir_sx and game.check_collision(point_l)) or
            (dir_up and game.check_collision(point_u)) or
            (dir_down and game.check_collision(point_d)),

            (dir_up and game.check_collision(point_r)) or  # Dx
            (dir_down and game.check_collision(point_l)) or
            (dir_sx and game.check_collision(point_u)) or
            (dir_dx and game.check_collision(point_d)),

            (dir_down and game.check_collision(point_r)) or  # Sx
            (dir_up and game.check_collision(point_l)) or
            (dir_dx and game.check_collision(point_u)) or
            (dir_sx and game.check_collision(point_d)),

            dir_sx,
            dir_dx,
            dir_up,
            dir_down,

            game.food.x < game.head.x,  # Food a Sx
            game.food.x > game.head.x,  # Food a Dx
            game.food.y < game.head.y,  # Food Up
            game.food.y > game.head.y  # Food Down
        ]
        return np.array(state, dtype=int) # Restituisco lo stato come array np

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Pop dell'item a sx se si raggiunge MAX_MEMORY

    def long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # estraggo un campione di tuple
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.q_trainer.train(states, actions, rewards, next_states, dones)

    def short_memory(self, state, action, reward, next_state, done):
        self.q_trainer.train(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games #TODO provare con epsilon diversi
        last_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            last_move[move] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.deep_model(state_0)
            move = torch.argmax(prediction).item()
            last_move[move] = 1

        return last_move
