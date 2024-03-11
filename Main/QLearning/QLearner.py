from random import random

import numpy as np


class ReplayBuffer:  # PER
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # Rimozione della meno recente se il buffer è pieno
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_best_experiences(self, top_n):
        # Ordinamento delle esperienze in base alla ricompensa totale e restituisci le prime top_n esperienze
        sorted_experiences = sorted(self.buffer, key=lambda x: x[2], reverse=True)  # x[2] è l'indice della ricompensa
        return sorted_experiences[:top_n]

    def reset_buffer(self):
        self.buffer = []


class QLearner:

    def __init__(self, state_space_size, action_space, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min, b_size):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.action_space_size = len(action_space)
        self.ql_table = np.zeros((self.state_space_size, self.action_space_size), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.keys = []
        self.key = 0
        self.replay_buffer = ReplayBuffer(b_size)

    def epsilon_greedy_policy(self, state):  # Scelta dell'azione secondo la policy epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.ql_table[self.find_key(state)])

    def update_ql_table(self, state, action, reward, next_state):

        max_next_q_value = np.max(self.ql_table[self.find_key(next_state)])  # Scelta Greedy dal next_state

        actual_q_value = self.ql_table[self.find_key(state), action]
        new_q_value = actual_q_value + self.alpha * (reward + self.gamma * max_next_q_value - actual_q_value)

        self.ql_table[self.find_key(state), action] = new_q_value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def find_key(self, state):
        if not (state in self.keys):
            self.keys.append(state)
            self.key += 1
            return self.key - 1
        else:
            i = 0
            while i < len(self.keys):
                if self.keys[i] == state:
                    return i
                i += 1

    def add_experience_to_replay_buffer(self, experience):
        state, action, reward, next_state, done = experience
        # Calcolo della ricompensa totale dell'episodio
        total_reward = sum(e[2] for e in self.replay_buffer.buffer) + reward
        experience_with_total_reward = (state, action, total_reward, next_state, done)
        self.replay_buffer.add_experience(experience_with_total_reward)

    def sample_batch_from_replay_buffer(self, batch_size, use_best_experiences):
        if use_best_experiences:
            best_experiences = self.replay_buffer.get_best_experiences(batch_size)
            return best_experiences
        else:
            return self.replay_buffer.sample_batch(batch_size)

    def reset_buffer(self):
        self.replay_buffer.reset_buffer()
