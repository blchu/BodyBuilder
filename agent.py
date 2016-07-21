import argparse
import gym
import logging
import sys

from dqn import DQN

log = logging.getLogger(__name__)

# number of episodes to train the agent for
NUM_EPISODES = 10000
# number of random actions taken for initialization
INIT_STEPS = 1000


algorithms = {
    'dqn': DQN
    }

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
    for _ in iterations:
        # step environment with random action and fill replay memory
        old_observation = observation
        observation, reward, done, _ = env.step(env.action_space.sample())

        # add state transition to replay memory
        network.notify_state_transition(old_observation, action, reward, observation, done)

        # reset the environment if done
        if done:
            observation = env.reset()

def train_agent(env, network):
    # train for NUM_EPISODES number of episodes
    current_episode = 0
    training_iterations = 0
    observation = env.reset()
    while current_episode < NUM_EPISODES:
        # render the environment
        if render:
            env.render()

        # take an action and step the environment
        action = network.training_predict(observation)
        old_observation = observation
        observation, reward, done, _ = env.step(action)

        # update network with state transition and train
        network.notify_state_transition(old_observation, action, reward, observation, done)
        network.batch_train()

        # reset the environment and start new episode if done
        if done:
            if render:
                env.render()
            observation = env.reset()
            current_episode += 1

def main():
    # initialize logging and parse command line flag arguments
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    
    # initialize network and prepare for training
    network = algorithms[network_algorithm](replay_memory)
    initialize_training(gym.make(env_name), network, INIT_STEPS)

    # begin training
    env = gym.make(env_name)
    train_agent(env, network)
    

if __name__ == '__main__':
    main()
