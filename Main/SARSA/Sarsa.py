import numpy as np

# Classe creata partendo dal QLearner e modificando la funzione di update_table
class Sarsa:

    def __init__(self, state_space_size, action_space, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.action_space_size = len(action_space)
        self.sarsa_table = np.zeros((self.state_space_size, self.action_space_size), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.keys = []
        self.key = 0

    def epsilon_greedy_policy(self, state):  # Scelta dell'azione secondo la policy epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.sarsa_table[self.find_key(state)])

    def update_sarsa_table(self, state, action, reward, next_state):

        next_state_value = self.sarsa_table[self.find_key(next_state)][self.epsilon_greedy_policy(next_state)]
        # Scelta epsilon-Greedy dal next_state

        actual_q_value = self.sarsa_table[self.find_key(state), action]
        new_q_value = actual_q_value + self.alpha * (reward + self.gamma * next_state_value - actual_q_value)

        self.sarsa_table[self.find_key(state), action] = new_q_value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def find_key(self, state):
        if not(state in self.keys):
            self.keys.append(state)
            self.key += 1
            return self.key - 1
        else:
            i = 0
            while i < len(self.keys):
                if self.keys[i] == state:
                    return i
                i += 1
