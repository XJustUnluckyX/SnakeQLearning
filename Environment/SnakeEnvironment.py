import math

import gym
import pygame
import numpy as np
from gym import spaces
from gym.spaces import Discrete


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size, render_mode):
        super(SnakeEnv, self).__init__()
        self.size = grid_size
        self.window_size = 512
        self.body = self._init_body()  # Il body è una tupla di dimensione 25, in cui la cella più vicina alla testa è
        # in prima posizione della tupla
        self.head = self._spawn_head()  # Generiamo casualmente la posizione di inizio
        self.action_space = spaces.Discrete(4)  # Mosse disponibili: 0: Su, 1: Giù, 2: Sx, 3: Dx
        self.observation_space = spaces.Dict(
            {
                "head": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
                "body": spaces.Box(low=-1, high=self.size - 1, shape=(2 * (self.size - 1),), dtype=int),
                "food": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int)
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
            self.head[0] -= 1
        elif action == 1:  # Giù
            self.head[0] += 1
        elif action == 2:  # Sx
            self.head[1] -= 1
        elif action == 3:  # Dx
            self.head[1] += 1
        self.last_action = action

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
        self.snake_length = 1
        for x in self.body:
            if x != [-1, -1]:
                self.snake_length = + 1

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

    def _init_body(self):
        body = tuple([-1, -1] for _ in range(self.size - 1))
        return body

    def _get_obs(self):
        return {"head": self.head, "body": self.body, "food": self.food}

    def _get_info(self):
        head_np = np.array(self.head)
        food_np = np.array(self.food)
        return {"distance": np.linalg.norm(head_np - food_np, ord=2), "snake_size": self.snake_length}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.food = []
        self.head = self._spawn_head()
        self.body = self._init_body()
        self.food = self._generate_food()
        self._update_snake_length()
        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def _update_body(self, previous_head):  # Aggiorna i body in modo da traslare tutte le celle di uno e di rimuovere
        # l'ultima cella diversa dal valore [-1, -1]
        for i in range(self.size - 1, 1, -1):
            j = i - 1
            self.body[i] = self.body[j]
        self.body[0] = previous_head

        for i in range(self.size - 2):
            if self.body[i] != [-1, -1] and self.body[i + 1] == [-1, -1]:
                self.body[i] = [-1, -1]
                return

        return

    def _increase_body(self, previous_head):
        for i in range(self.size - 1, 1, -1):
            j = i - 1
            self.body[i] = self.body[j]
        self.body[0] = previous_head

    def step(self, action):
        previous_head = self.head
        self._update_snake_position(action)

        done = False
        if self._check_collision():
            reward = -1000  # Penalità per collisione
            done = True
        elif self.head == self.food:
            reward = 10  # Ricompensa per mangiare il cibo
            self.food = self._generate_food()
            self._increase_body(previous_head)
        else:
            # Ricompensa intermedia per avvicinamento al cibo
            reward = -1 * math.dist(self.head, self.food)
            if self.snake_length != 1:
                self._update_body(previous_head)

        self._update_snake_length()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, done, False, self._get_info()

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
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

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

        # body != [-1, -1]
        for i in range(self.size - 1):
            if self.body[i] != [-1, -1]:
                pygame.draw.rect(
                    canvas,
                    (255, 0, 0),
                    pygame.Rect(
                        int(pix_square_size * self.body[i][0]),
                        int(pix_square_size * self.body[i][1]),
                        int(pix_square_size),
                        int(pix_square_size),
                    ),
                )

        # food
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int((self.food[0] + 0.5) * pix_square_size),
             int((self.food[1] + 0.5) * pix_square_size)),
            int(pix_square_size / 3),
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
