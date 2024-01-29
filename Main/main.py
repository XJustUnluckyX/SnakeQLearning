import gym
from gym.wrappers import FlattenObservation

env = gym.make("SnakeQl")
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())
