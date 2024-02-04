import gym.envs.registration
import QLearning.QLearner as Ql
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
with open('q_learner.pkl', 'rb') as input_file:
    best_q_learner = pickle.load(input_file)
    print("QLearner table:")
    print(best_q_learner.ql_table)
    print("QLearner epsilon:")
    print(best_q_learner.epsilon)
    print("QLearner epsilon min:")
    print(best_q_learner.epsilon_min)
    print("QLearner epsilon decay:")
    print(best_q_learner.epsilon_decay)
    print("QLearner alpha:")
    print(best_q_learner.alpha)
    print("QLearner gamma:")
    print(best_q_learner.gamma)
    num_episodes = 5000
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
            action = best_q_learner.epsilon_greedy_policy(int_state)
            next_state, reward, done, length, info = env.step(action)
            str_next_state = re.sub("[^0-9]", "", str(next_state))
            int_next_state = int(str_state)
            # Modifica dell'aggiornamento della Q-table con ricompensa intermedia
            best_q_learner.update_ql_table(int_state, action, reward, int_next_state)

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
                plot(plot_scores, plot_mean_scores)
                break

        # Aggiornamento del tasso di esplorazione
        best_q_learner.decay_epsilon()

    print(f"Max reward: {max_reward}, relative length: {temp_max_length}")
    with open('q_learner.pkl', 'wb') as output:
        pickle.dump(best_q_learner, output, pickle.HIGHEST_PROTOCOL)