import gym.envs.registration
import QLearning.QLearner as Ql
import re

gym.register(
    id='SnakeQl',
    entry_point='Environment.SnakeEnvironment:SnakeEnv',
    max_episode_steps=10000,
)

# Parametri dell'ambiente Snake
grid_size = 5

env = gym.make("SnakeQl", grid_size=grid_size, render_mode="human")


observation_space_size = ((grid_size * grid_size) * (grid_size * grid_size)) * (5 * ((grid_size * grid_size) - 1))
# Possibili posizioni per la Snake head * possibili
# posizioni per il cibo * possibili valori delle grid_size^2 -1 azioni

print(observation_space_size)
action_space = [0, 1, 2, 3]

# Grid Search
learning_rate = [0.1, 0.2]
gamma = [0.7, 0.5]
epsilon_start = [0.6, 0.4]
epsilon_decay = [0.995, 0.9]
epsilon_min = [0.03, 0.05, 0.1]
replay_buffer_size = [1000, 2000]
best_parameters = []

max_reward_grid_search = -100000
rel_length = 0

for a in range(len(learning_rate)):
    for b in range(len(gamma)):
        for c in range(len(epsilon_start)):
            for d in range(len(epsilon_decay)):
                for e in range(len(epsilon_min)):
                    for f in range(len(replay_buffer_size)):

                        # Creazione dell'agente Q-learning con i parametri della Grid Search
                        q_learner = Ql.QLearner(observation_space_size, action_space, learning_rate[a],
                                                gamma[b], epsilon_start[c], epsilon_decay[d], epsilon_min[e])

                        # Parametri di addestramento
                        num_episodes = 1000
                        max_reward = -10000
                        length = 0
                        temp_max_length = 0
                        # Ciclo di addestramento
                        for episode in range(num_episodes):
                            state, info = env.reset()

                            total_reward = 0

                            while True:
                                #Conversione dello stato in un intero, per poter definire la riga corrispondente nella
                                #tabella del Q-learning
                                str_state = re.sub("[^0-9]", "", str(state))
                                int_state = int(str_state)
                                action = q_learner.epsilon_greedy_policy(int_state)
                                next_state, reward, done, length, _ = env.step(action)
                                str_next_state = re.sub("[^0-9]", "", str(next_state))
                                int_next_state = int(str_state)
                                # Modifica dell'aggiornamento della Q-table con ricompensa intermedia
                                q_learner.update_ql_table(int_state, action, reward, int_next_state)

                                total_reward += reward
                                state = next_state
                                if max_reward < total_reward:
                                    max_reward = total_reward
                                    temp_max_length = length

                                if done:
                                    # print(f"Episodio: {episode}, Ricompensa totale: {total_reward}")
                                    break

                            # Aggiornamento del tasso di esplorazione
                            q_learner.decay_epsilon()

                        print(f"Max reward: {max_reward}, last length: {length}")
                        if max_reward_grid_search < max_reward:
                            max_reward_grid_search = max_reward
                            rel_length = temp_max_length
                            best_parameters = [learning_rate[a], gamma[b], epsilon_start[c], epsilon_decay[d],
                                               epsilon_min[e], replay_buffer_size[f]]
                            print(f"New best parameters: {best_parameters}")

print(f"Max Reward Grid Search: {max_reward_grid_search}, Relative Length: {rel_length}")
print(f"Best parameters: {best_parameters}")
