from q_learner import QLearner
from snake_env import SnakeEnv
from plotter import plot

# E' stato utilizzato come riferimento il seguente progetto
# https://github.com/patrickloeber/snake-ai-pytorch/tree/main , poi rifattorizzato

# Grid Search
learning_rate = [0.001, 0.01, 0.1]
gamma = [0.9, 0.7, 0.5]
epsilon_start = [0.8, 0.6, 0.4]
epsilon_decay = [0.95, 0.995, 0.9]
epsilon_min = [0.03, 0.05, 0.1, 0.05]
batch_size = [1000, 2000]
best_parameters = []

max_reward_grid_search = -100000
rel_length = 0

for a in range(len(learning_rate)):
    for b in range(len(gamma)):
        for c in range(len(epsilon_start)):
            for d in range(len(epsilon_decay)):
                for e in range(len(epsilon_min)):
                    for f in range(len(batch_size)):
                        env = SnakeEnv()
                        # Parametri di addestramento
                        num_episodes = 500
                        max_reward = -10000
                        length = 0
                        temp_max_length = 0
                        total_score = 0
                        total_reward = 0
                        plot_scores = []
                        plot_mean_scores = []
                        record = -1
                        agent = QLearner(epsilon_start[c], gamma[b], learning_rate[a], batch_size[f], epsilon_decay[d],
                                         epsilon_min[e])
                        # Ciclo di addestramento
                        for episode in range(num_episodes):
                            env.respawn_snake()
                            while True:
                                state_old = agent.compute_state(env)
                                last_move = agent.pred_action(state_old)
                                reward, done, score = env.step(last_move)
                                state_new = agent.compute_state(env)
                                agent.learn_with_short_memory(state_old, last_move, reward, state_new, done)
                                agent.add_to_memory(state_old, last_move, reward, state_new, done)

                                if done:
                                    env.respawn_snake()
                                    agent.num_games += 1
                                    agent.learn_with_long_memory()
                                    if score > record:  # Salvo le migliori partite
                                        record = score
                                        agent.deep_model.save_model()

                                    print('Game', agent.num_games, 'Score', score, 'Record:', record)

                                    total_score += score
                                    mean_score = total_score / agent.num_games
                                    plot_scores.append(score)
                                    plot_mean_scores.append(mean_score)
                                    plot(plot_scores, plot_mean_scores)
                                    break

print(f"Max Reward Grid Search: {max_reward_grid_search}, Relative Length: {rel_length}")
print(f"Best parameters: {best_parameters}")
