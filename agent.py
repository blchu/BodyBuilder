import argparse
import gym
import sys

from dqn import DQN

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
    # train for NUM_EPISODES number of episodes
    current_episode = 0
    training_iterations = 0
    observation = env.reset()
    while current_episode < NUM_EPISODES:
        # render the environment
        if render:
            env.render()

        # take an action and step the environment
        action = network.training_predict(env, observation)
        old_observation = observation
        observation, reward, done, _ = env.step(action)

        # update network with state transition and train
        network.notify_state_transition(old_observation, action, reward, done)
        network.batch_train()

        # reset the environment and start new episode if done
        if done:
            network.notify_terminal()
            if render:
                env.render()
            observation = env.reset()
            current_episode += 1

def main():
    # parse command line flag arguments
    args = parse_arguments()
    
    # initialize network and prepare for training
    network = algorithms[network_algorithm]()
    initialize_training(gym.make(env_name), network, INIT_STEPS)

    # begin training
    env = gym.make(env_name)
    train_agent(env, network)
    

if __name__ == '__main__':
    main()
