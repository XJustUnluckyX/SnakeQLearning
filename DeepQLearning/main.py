from learner import QLearner
from env import Env
from plotter import plot

plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0
agent = QLearner()
env = Env()

while True:
    state_old = agent.get_state(env)
    final_move = agent.get_action(state_old)
    reward, done, score = env.step(final_move)
    state_new = agent.get_state(env)
    agent.short_memory(state_old, final_move, reward, state_new, done)
    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
        env.reset()
        agent.n_games += 1
        agent.long_memory()
        if score > record:  # Salvo le migliori partite
            record = score
            agent.deep_model.save_model()

        print('Game', agent.n_games, 'Score', score, 'Record:', record)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
