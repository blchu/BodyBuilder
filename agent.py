import argparse
import gym
import select
import sys
import threading
import time

from dqn import DQN
from enums import EnvTypes

# number of episodes to train and test the agent for
TRAIN_EPISODES = 3000
TEST_EPISODES = 1000
# number of random actions taken for initialization
INIT_STEPS = 10000

atari_environments = {
    'Breakout-v0': 6,
    'MsPacman-v0': 8,
    'Phoenix-v0': 8,
    'SpaceInvaders-v0': 6
    }

standard_environments = {
    'CartPole-v0': [4, 2]
    }

algorithms = {
    'dqn': DQN
    }

render = True
polling = True

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
        if render:
            env.render()

        observation, reward, done, _ = env.step(network.testing_predict(observation))
        tot_reward += reward

        if done:
            print("Episode {} completed; total reward is {}".format(curr_episode, tot_reward))
            if render:
                env.render()
            observation = env.reset()
            curr_episode += 1
            tot_reward = 0

def render_toggle():
    global render
    while(polling):
        register = select.select([sys.stdin], [], [], 0.1)[0]
        if len(register) and register[0].readline():
            render = not render

def main():
    global polling

    # parse command line flag arguments
    args = parse_arguments()

    # currently only support certain environments
    assert (args.env_name in atari_environments.keys() or 
            args.env_name in standard_environments.keys())

    # add keyboard polling for render toggling
    render_toggle_thread = threading.Thread(target=render_toggle)
    render_toggle_thread.start()

    try:
        if args.env_name in atari_environments.keys():
            env_type = EnvTypes.ATARI
            state_dims = [105, 80, 1]
            action_dims = atari_environments[args.env_name]
        elif args.env_name in standard_environments.keys():
            env_type = EnvTypes.STANDARD
            state_dims = [standard_environments[args.env_name][0]]
            action_dims = standard_environments[args.env_name][1]
        
        # initialize network and prepare for training
        network = algorithms[args.network_algorithm](env_type, state_dims, action_dims)
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

    except KeyboardInterrupt:
        print("\nInterrupt received. Terminating agent...")

    finally:
        # stop the keyboard polling thread
        polling = False
        render_toggle_thread.join()
        sys.exit()


if __name__ == '__main__':
    main()
