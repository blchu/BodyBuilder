import gym
import sys
import time
import agent
from dqn import DQN
from enums import EnvTypes

def test_replay_memory_sampling():
    env = gym.make('Breakout-v0')
    dqn = DQN(EnvTypes.ATARI, [105, 80], 6)
    print("Filling replay memory...")
    agent.initialize_training(env, dqn, dqn.replay_memory.capacity)
    print("Testing sampling performance...")
    t = time.time()
    for _ in range(10000):
        dqn.replay_memory.sample(64)
    print("Total sampling time for 10000 iterations is {} seconds".format(time.time() - t))

