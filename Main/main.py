from gym.wrappers import FlattenObservation
import gym.envs.registration

gym.register(
    id='SnakeQl',
    entry_point='Environment.SnakeEnvironment:SnakeEnv',
    max_episode_steps=10000,
)

env = gym.make("SnakeQl", grid_size=10, render_mode="human")
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())
