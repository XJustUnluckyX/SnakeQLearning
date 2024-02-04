import gym.envs.registration
import SARSA.Sarsa as Sarsa
import re
from plotter import plot
import pickle

gym.register(
    id='SnakeQl',
    entry_point='Environment.SnakeEnvironment:SnakeEnv',
    max_episode_steps=10000,
)

# Parametri dell'ambiente Snake
grid_size = 5

env = gym.make("SnakeQl", grid_size=grid_size, render_mode="human")

observation_space_size = ((grid_size * grid_size) * 2 * 2) * (5 ** grid_size)
# Possibili posizioni per la Snake head * possibili
# posizioni per il cibo * possibili valori delle grid_size^2 -1 azioni

action_space = [0, 1, 2, 3]
print(observation_space_size)
# Grid Search
alpha = [0.5, 0.3]
gamma = [0.7, 0.5]
epsilon_start = [0.5, 0.8, 0.6]
epsilon_decay = [0.995, 0.9]
epsilon_min = [0.03, 0.05]
best_parameters = []

max_reward_grid_search = -100000
rel_length = 0
best_sarsa_learner = 0
for a in range(len(alpha)):
    for b in range(len(gamma)):
        for c in range(len(epsilon_start)):
            for d in range(len(epsilon_decay)):
                for e in range(len(epsilon_min)):

                    # Creazione dell'agente Q-learning con i parametri della Grid Search
                    sarsa = Sarsa.Sarsa(observation_space_size, action_space, alpha[a],
                                        gamma[b], epsilon_start[c], epsilon_decay[d], epsilon_min[e])

                    # Parametri di addestramento
                    num_episodes = 1000
                    max_reward = -10000
                    total_score = 0
                    total_reward = 0
                    plot_scores = []
                    plot_mean_scores = []
                    n_games = 0
                    temp_max_length = 0
                    # Ciclo di addestramento
                    for episode in range(num_episodes):
                        state, info = env.reset()
                        length = 0
                        total_reward = 0

                        while True:
                            # Conversione dello stato in un intero, per poter definire la riga corrispondente nella
                            # tabella del Q-learning
                            str_state = re.sub("[^0-9]", "", str(state))
                            int_state = int(str_state)
                            action = sarsa.epsilon_greedy_policy(int_state)
                            next_state, reward, done, length, info = env.step(action)
                            str_next_state = re.sub("[^0-9]", "", str(next_state))
                            int_next_state = int(str_state)
                            # Modifica dell'aggiornamento della Q-table con ricompensa intermedia
                            sarsa.update_sarsa_table(int_state, action, reward, int_next_state)

                            total_reward += reward
                            state = next_state

                            if done:
                                n_games += 1
                                if max_reward < total_reward:
                                    max_reward = total_reward
                                    temp_max_length = length
                                total_score += (length - 1)
                                mean_score = total_score / n_games
                                plot_scores.append(length - 1)
                                plot_mean_scores.append(mean_score)
                                #plot(plot_scores, plot_mean_scores)
                                break

                        # Aggiornamento del tasso di esplorazione
                        sarsa.decay_epsilon()

                    print(f"Max reward: {max_reward}, relative length: {temp_max_length}")
                    if max_reward_grid_search < max_reward:
                        max_reward_grid_search = max_reward
                        rel_length = temp_max_length
                        best_parameters = [alpha[a], gamma[b], epsilon_start[c], epsilon_decay[d],
                                           epsilon_min[e]]
                        best_sarsa_learner = sarsa
                        print(f"New best parameters: {best_parameters}")

print(f"Max Reward Grid Search: {max_reward_grid_search}, Relative Length: {rel_length}")
print(f"Best parameters: {best_parameters}")
with open('sarsa_learner.pkl', 'wb') as output_file:
    pickle.dump(best_sarsa_learner, output_file, pickle.HIGHEST_PROTOCOL)