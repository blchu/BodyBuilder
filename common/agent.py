import gym

NUM_EPISODES = 10000

def train_agent(env, network, replay_memory, render=True):
    # train for NUM_EPISODES number of episodes
    current_episode = 0
    observation = env.reset()
    while current_episode < NUM_EPISODES:
        # render the environment
        if render:
            env.render()

        # take an action and step the environment
        action = network.predict(observation)
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
