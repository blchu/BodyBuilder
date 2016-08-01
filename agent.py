import argparse
import gym
import sys

from dqn import DQN
from enums import EnvTypes

# number of episodes to train and test the agent for
TRAIN_EPISODES = 2000
TEST_EPISODES = 1000
# number of random actions taken for initialization
INIT_STEPS = 10000

atari_environments = {
    'Breakout-v0': 6,
    'MsPacman-v0': 8,
    'Phoenix-v0': 8,
    'SpaceInvaders-v0': 6
    }

algorithms = {
    'dqn': DQN
    }

render = False #True

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('env_name',
                        help="Name of the OpenAI Gym environment to run")
    parser.add_argument('network_algorithm',
                        help="The algorithm to be trained")
    parser.add_argument('--monitor', default=None,
                        help="Directory for monitor recording of training")
    parser.add_argument('--initsteps', default=INIT_STEPS,
                        help="Number of steps taken during initialization")

    return parser.parse_args()

def initialize_training(env, network, iterations):
    observation = env.reset()
    for i in range(iterations):
        # step environment with random action and fill replay memory
        old_observation = observation
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        # add state transition to replay memory
        network.process_observation(old_observation)
        network.notify_state_transition(action, reward, done)

        # reset the environment if done
        if done:
            observation = env.reset()

        # show progress every thousand steps
        step = i+1
        if step % 1000 == 0:
            print("Training initialization step {} completed".format(step))

def train_agent(env, network):
    print("Beginning training")
    # train for NUM_EPISODES number of episodes
    curr_episode = 0
    training_iterations = 0
    tot_reward = 0
    observation = env.reset()
    while curr_episode < TRAIN_EPISODES:
        # render the environment
        if render:
            env.render()

        # take an action and step the environment
        action = network.training_predict(env, observation)
        old_observation = observation
        observation, reward, done, _ = env.step(action)
        tot_reward += reward

        # update network with state transition and train
        network.notify_state_transition(action, reward, done)
        network.batch_train()

        # reset the environment and start new episode if done
        if done:
            print("Episode {} completed; total reward is {}".format(curr_episode, tot_reward))
            if render:
                env.render()
            observation = env.reset()
            curr_episode += 1
            tot_reward = 0

        # display training iterations every 10 iterations
        training_iterations += 1
        if training_iterations % 10 == 0:
            print("Agent training iteration {} completed".format(training_iterations))

def test_agent(env, network):
    print("beginning testing")
    # test for TEST_EPISODES number of episodes
    curr_episode = 0
    tot_reward = 0
    observation = env.reset()
    while curr_episode < TEST_EPISODES:
        env.render()
        observation, _, done, _ = env.step(network.testing_predict(observation))

        if done:
            print("Episode {} completed; total reward is {}".format(curr_episode, tot_reward))
            env.render()
            observation = env.reset()
            curr_episode += 1
            tot_reward = 0

def main():
    # parse command line flag arguments
    args = parse_arguments()

    # currently only support certain atari environments
    assert args.env_name in atari_environments.keys()
    
    # initialize network and prepare for training
    network = algorithms[args.network_algorithm](EnvTypes.ATARI, [105, 80],
                                                 atari_environments[args.env_name])
    initialize_training(gym.make(args.env_name), network, INIT_STEPS)

    # begin training
    train_env = gym.make(args.env_name)
    if args.monitor is not None:
        train_env.monitor.start(args.monitor+'/train')
    train_agent(train_env, network)
    if args.monitor is not None:
        train_env.monitor.close()

    # evaluate agent
    test_env = gym.make(args.env_name)
    if args.monitor is not None:
        test_env.monitor.start(args.monitor+'/test')
    test_agent(test_env, network)
    if args.monitor is not None:
        test_env.monitor.close()
    

if __name__ == '__main__':
    main()
