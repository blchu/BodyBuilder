import os
import random
import shutil

import numpy as np
import tensorflow as tf

from enums import EnvTypes
from ops import conv2d, fc, mse
from replay_memory import ReplayMemory
from utils import rgb_to_luminance, downscale

FRAME_STACK = 4
FRAME_SKIP = 4

REPLAY_MEMORY_CAPACITY = 50000  # fifty thousand

# hyperparameters
ALPHA = 25e-5             # initial learning rate
GAMMA = 0.9               # discount factor
EPSILON = 1e-2            # numerical stability
RMS_DECAY = 0.95          # rmsprop decay
MOMENTUM = 0.95           # rmsprop momentum
FINAL_EXPLORATION = 0.1   # final exploration rate
EXPLORATION_DECAY = 1e-6  # linear decay of exploration
BATCH_SIZE = 64           # size of training batch
TARGET_UPDATE = 1000      # iterations per target network update

# scope names
DQN_SCOPE = 'dqn'
TARGET_SCOPE = 'target'
TRANSFER_SCOPE = 'transfer'
EVALUATION_SCOPE = 'evaluator'

# layer names
CONV1 = 'conv1'
CONV2 = 'conv2'
CONV3 = 'conv3'
HIDDEN = 'hidden'
OUTPUT = 'output'

TENSORBOARD_GRAPH_DIR = "/tmp/dqn"

GPU_MEMORY_FRACTION = 0.5

class DQN():

    def __init__(self, env_type, state_dims, num_actions):
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_CAPACITY,
                                          [state_dims[0], state_dims[1]*FRAME_STACK],
                                          num_actions)
        self.exploration = 1.0
        self.train_iter = 0

        if env_type == EnvTypes.ATARI:
            buffer_size = FRAME_STACK*FRAME_SKIP
            self.observation_buffer = [np.zeros((state_dims[0], state_dims[1]))
                                       for _ in range(buffer_size)]

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        self.sess = tf.Session(config=self.config)

        # build network
        self.dqn_vars = dict()
        with tf.variable_scope(DQN_SCOPE):
            if env_type == EnvTypes.ATARI:
                self.x = tf.placeholder(tf.float32, shape=[None, state_dims[0],
                                                           state_dims[1]*FRAME_STACK, 1])
                self.conv1 = conv2d(self.x, 8, 4, 32, CONV1, var_dict=self.dqn_vars)
                self.conv2 = conv2d(self.conv1, 4, 2, 64, CONV2, var_dict=self.dqn_vars)
                self.conv3 = conv2d(self.conv2, 3, 1, 64, CONV3, var_dict=self.dqn_vars)
                conv_shape = self.conv3.get_shape().as_list()
                flatten = [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]]
                self.initial_layers = tf.reshape(self.conv3, flatten)

            # add final hidden layers
            self.hid = fc(self.initial_layers, 512, HIDDEN, var_dict=self.dqn_vars)
            self.q = fc(self.hid, num_actions, OUTPUT,
                        var_dict=self.dqn_vars, activation=False)
                          
        # build target network
        self.target_vars = dict()
        with tf.variable_scope(TARGET_SCOPE):
            if env_type == EnvTypes.ATARI:
                self.t_x = tf.placeholder(tf.float32, shape=[None, state_dims[0],
                                                             state_dims[1]*FRAME_STACK, 1])
                self.t_conv1 = conv2d(self.t_x, 8, 4, 32, CONV1, var_dict=self.target_vars)
                self.t_conv2 = conv2d(self.t_conv1, 4, 2, 64, CONV2, var_dict=self.target_vars)
                self.t_conv3 = conv2d(self.t_conv2, 3, 1, 64, CONV3, var_dict=self.target_vars)
                conv_shape = self.t_conv3.get_shape().as_list()
                flatten = [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]]
                self.t_initial_layers = tf.reshape(self.t_conv3, flatten)

            self.t_hid = fc(self.t_initial_layers, 512, HIDDEN, var_dict=self.target_vars)
            self.t_q = fc(self.t_hid, num_actions, OUTPUT,
                          var_dict=self.target_vars, activation=False)

        # add weight transfer operations from primary dqn network to target network
        self.assign_ops = []
        with tf.variable_scope(TRANSFER_SCOPE):
            for variable in self.dqn_vars.keys():
                target_variable = TARGET_SCOPE + variable[len(DQN_SCOPE):]
                target_assign = self.target_vars[target_variable].assign(self.dqn_vars[variable])
                self.assign_ops.append(target_assign)

        # build dqn evaluation
        with tf.variable_scope(EVALUATION_SCOPE):
            # one-hot action selection
            self.action = tf.placeholder(tf.float32, shape=[None, num_actions])
            # reward
            self.reward = tf.placeholder(tf.float32, shape=[None, 1])
            # terminal state
            self.non_terminal = tf.placeholder(tf.float32, shape=[None, 1])

            self.target = tf.add(self.reward, tf.mul(GAMMA, tf.mul(self.non_terminal,
                          tf.reduce_max(self.t_q, 1, True))))
            self.predict = tf.reduce_sum(tf.mul(self.action, self.q), 1, True)
            self.error = mse(self.predict, self.target)
        
        self.optimize = tf.train.RMSPropOptimizer(ALPHA, decay=RMS_DECAY, momentum=MOMENTUM,
                        epsilon=EPSILON).minimize(self.error, var_list=self.dqn_vars.values())

        # initialize variables
        self.sess.run(tf.initialize_all_variables())
        
        # write out the graph for tensorboard
        if os.path.isdir(TENSORBOARD_GRAPH_DIR):
            shutil.rmtree(TENSORBOARD_GRAPH_DIR)
        self.writer = tf.train.SummaryWriter(TENSORBOARD_GRAPH_DIR, self.sess.graph)

    def process_observation(self, observation):
        # convert to normalized luminance and downscale
        observation = downscale(rgb_to_luminance(observation), 2)

        # push the new observation onto the buffer
        self.observation_buffer.pop(len(self.observation_buffer)-1)
        self.observation_buffer.insert(0, observation)

    def _get_stacked_state(self):
        stacked_state = self.observation_buffer[0]
        for i in range(1, FRAME_STACK):
            stacked_state = np.hstack((stacked_state, self.observation_buffer[i*FRAME_SKIP]))
        return stacked_state

    def _predict(self):
        print("Running prediction")
        stacked_state = self._get_stacked_state()
        stacked_state = np.expand_dims(stacked_state, axis=0)
        stacked_state = np.expand_dims(stacked_state, axis=3)
        return np.argmax(self.sess.run(self.q, feed_dict={self.x: stacked_state}))

    def training_predict(self, env, observation):
        self.process_observation(observation)

        # select action according to epsilon-greedy policy
        if random.random() < self.exploration:
            action = env.action_space.sample()
        else:
            action = self._predict()
        self.exploration = max(self.exploration - EXPLORATION_DECAY, FINAL_EXPLORATION)

        return action

    def testing_predict(self, observation):
        self.process_observation(observation)
        return self._predict()

    def notify_state_transition(self, action, reward, done):
        state = self._get_stacked_state()
        self.replay_memory.add_state_transition(state, action, reward, done)
        if done:
            # flush the observation buffer
            for i in range(len(self.observation_buffer)):
                self.observation_buffer[i] = np.zeros(self.observation_buffer[i].shape)

    def batch_train(self):
        # sample batch from replay memory
        states, actions, rewards, terminals, newstates = self.replay_memory.sample(BATCH_SIZE)
        states = np.expand_dims(states, axis=3)
        newstates = np.expand_dims(newstates, axis=3)
        rewards = np.expand_dims(rewards, axis=1)
        terminals = np.expand_dims(terminals, axis=1)
        nonterminals = 1 - terminals

        # update target network weights
        if self.train_iter % TARGET_UPDATE == 0:
            self.sess.run(self.assign_ops)

        # run neural network training step
        self.sess.run(self.optimize, feed_dict={self.x:states, self.t_x:newstates,
                      self.action:actions, self.reward:rewards, self.non_terminal:nonterminals})

        self.train_iter += 1
