from gym.envs.registration import register


register(
    id='SnakeQl',
    entry_point='./SnakeQLearning.Environment:SnakeEnvironment',
    max_episode_steps=10000,
)
