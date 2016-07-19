import argparse
import gym
import logging
import sys

log = logging.getLogger(__name__)

# number of episodes to train the agent for
NUM_EPISODES = 10000
# initial replay memory size
FILL_REPLAY = 1000

# mandatory command line arguments
ENVIRONMENT = 'env_name'
ALGORITHM = 'network_algorithm'
# command line argument flags
MONITOR_DIR_FLAG = '--monitor'


algorithms = {
    'dqn': print #DQN
    }

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(ENVIRONMENT, action='store')
    parser.add_argument(ALGORITHM, action='store')
    parser.add_argument(MONITOR_DIR_FLAG, dest='monitor', help="Directory for monitor recording of training")

    return parser.parse_args()

def fill_memory(env, replay_memory, iterations):
    observation = env.reset()
    for _ in iterations:
        # step environment with random action and fill replay memory
        old_observation = observation
        observation, reward, done, _ = env.step(env.action_space.sample())

        # add state transition to replay memory
        replay_memory.add_state_transition(old_observation, action, reward, observation, done)

        # reset the environment if done
        if done:
            observation = env.reset()

def train_agent(env, network, replay_memory, render=True):
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

        # add state transition to replay memory
        replay_memory.add_state_transition(old_observation, action, reward, observation, done)

        # train the network
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
    
    # initialize replay memory and network
    replay_memory = ReplayMemory()
    network = algorithms[network_algorithm](replay_memory)

    # fill replay memory
    fill_memory(gym.make(env_name), replay_memory, FILL_REPLAY)

    # begin training
    

if __name__ == '__main__':
    main()
