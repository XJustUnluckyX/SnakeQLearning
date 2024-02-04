import math

import gym
import pygame
import numpy as np
from gym import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12000}

    def __init__(self, grid_size, render_mode):
        super(SnakeEnv, self).__init__()
        self.size = grid_size
        self.total_size = self.size * self.size
        self.window_size = 512
        self.body = []
        self.actions = self._init_actions()
        # in prima posizione della tupla
        self.head = self._spawn_head()  # Generiamo casualmente la posizione di inizio
        self.action_space = spaces.Discrete(4)  # Mosse disponibili: 0: Su, 1: Giù, 2: Sx, 3: Dx
        self.observation_space = spaces.Dict(  # Per ridurre lo spazio delle osservazioni, teniamo traccia delle
            # features degli stati
            {
                "head": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
                "actions": spaces.Box(low=-1, high=3, shape=(grid_size,), dtype=int),
                "food": spaces.Box(low=0, high=1, shape=(2,), dtype=int)
            }
        )

        self.food = self._generate_food()  # Generiamo casualmente la posizione del cibo, diversa da quella della testa
        self.last_action = -1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._update_snake_length()

        self.window = None
        self.clock = None

    def _update_snake_position(self, action):
        action = self._check_opposite(action)
        if action == 0:  # Su
            self.head[1] = self.head[1] + 1
        elif action == 1:  # Giù
            self.head[1] = self.head[1] - 1
        elif action == 2:  # Sx
            self.head[0] = self.head[0] - 1
        elif action == 3:  # Dx
            self.head[0] = self.head[0] + 1
        self.last_action = action

    def _update_food_direction(self):
        x1_bool = self.food[0] < self.head[0],  # Food a Sx o a Dx
        y1_bool = self.food[1] < self.head[1],  # Food Su o Giù
        directions = [x1_bool, y1_bool]
        return directions

    def _check_opposite(self, action):  # Controlla che l'azione non sia nel verso opposto dell'ultima azione eseguita
        if (self.last_action == 0) and (action == 1):
            action = self.last_action
        if (self.last_action == 1) and (action == 0):
            action = self.last_action
        if (self.last_action == 2) and (action == 3):
            action = self.last_action
        if (self.last_action == 3) and (action == 2):
            action = self.last_action
        return action

    def _generate_food(self):
        food_position = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        while food_position in [list(self.head)] + list(self.body):  # Controllo che il cibo non sia generato dove si
            # trova lo Snake
            food_position = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        return food_position

    def _spawn_head(self):
        head_position = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        return head_position

    def _update_snake_length(self):
        self.snake_length = len(self.body) + 1

    def _check_collision(self):
        if (
                self.head[0] < 0
                or self.head[0] >= self.size
                or self.head[1] < 0
                or self.head[1] >= self.size
                or self.head in self.body
        ):
            return True
        return False

    def _get_obs(self):
        food_direction = self._update_food_direction()
        return {"head": self.head, "actions": self.actions, "food": food_direction}

    def _get_info(self):
        head_np = np.array(self.head)
        food_np = np.array(self.food)
        return {"distance": np.linalg.norm(head_np - food_np, ord=2), "snake_size": self.snake_length}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.food = []
        self.head = self._spawn_head()
        self.body = []
        self.actions = self._init_actions()
        self.food = self._generate_food()
        self._update_snake_length()
        obs = self._get_obs()
        info = self._get_info()
        self._render_frame()
        return obs, info

    def _increase_actions(self, last_action):
        i = 4
        while i >= 1:
            j = i - 1
            self.actions[i] = self.actions[j]
            i -= 1
        self.actions[0] = last_action

    def step(self, action):
        previous_head = [self.head[0], self.head[1]]
        self._update_snake_position(action)
        self._increase_actions(action)
        self._update_snake_length()
        done = False
        if self._check_collision():
            reward = -100  # Penalità per collisione
            done = True
            self._render_frame()
            self.body.append([previous_head[0], previous_head[1]])  # La testa precedente diventa l'ultimo valore
            # della lista del body
            self.body.pop(0)
            self._update_snake_length()

            return self._get_obs(), reward, done, self.snake_length, self._get_info()
        if self.head == self.food:
            reward = 10  # Ricompensa per mangiare il cibo
            self._render_frame()
            self.body.append([previous_head[0], previous_head[1]])
            self.food = self._generate_food()
        else:
            # Ricompensa intermedia per avvicinamento al cibo
            center = self.size / 2
            centered_head = [self.head[0] / center, self.head[1] / center]
            centered_food = [self.food[0] / center, self.food[1] / center]
            reward = 0.1 * (2 - math.dist(centered_head, centered_food))
            self._render_frame()
            self.body.append([previous_head[0], previous_head[1]])  # La testa precedente diventa l'ultimo valore
            # della lista del body
            self.body.pop(0)
        self._render_frame()

        self._update_snake_length()
        return self._get_obs(), reward, done, self.snake_length, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
                self.window_size / self.size
        )

        # food
        if self.head != self.food:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (int((self.food[0] + 0.5) * pix_square_size),
                 int((self.food[1] + 0.5) * pix_square_size)),
                int(pix_square_size / 3),
            )
            # head
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    int(pix_square_size * self.head[0]),
                    int(pix_square_size * self.head[1]),
                    int(pix_square_size),
                    int(pix_square_size),
                ),
            )
        else:
            # head
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    int(pix_square_size * self.head[0]),
                    int(pix_square_size * self.head[1]),
                    int(pix_square_size),
                    int(pix_square_size),
                ),
            )

        # body
        body_green = 0
        for i in range(len(self.body)):
            body_green = min(body_green+(255/(len(self.body)+1)), 150)
            pygame.draw.rect(
                canvas,
                (0,body_green,0),
                pygame.Rect(
                    int(pix_square_size * self.body[i][0]),
                    int(pix_square_size * self.body[i][1]),
                    int(pix_square_size),
                    int(pix_square_size),
                ),
            )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _init_actions(self):
        actions = np.full(5, 5)
        return actions
