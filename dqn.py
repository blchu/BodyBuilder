import os
import random
import shutil

import numpy as np
import tensorflow as tf

from ops import conv2d, fc, mse
from replay_memory import ReplayMemory

FRAME_STACK = 4
FRAME_SKIP = 4

REPLAY_MEMORY_CAPACITY = 100000  # one hundred thousand

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

    def __init__(self, env_info):
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_CAPACITY, env_info['shape'],
                                          env_info['num_actions'])
        self.exploration = 1.0
        self.train_iter = 0

        if env_info['type'] == 'atari':
            shape = env_info['shape']
            buffer_size = FRAME_STACK*FRAME_SKIP
            self.observation_buffer = [np.zeros((shape[0], shape[1])) for _ in range(buffer_size)]

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        self.sess = tf.Session(config=self.config)

        # build network
        self.dqn_vars = dict()
        with tf.variable_scope(DQN_SCOPE):
            if env_info['type'] == 'atari':
                self.x = tf.placeholder(tf.float32, shape=[None, env_info['shape'][0],
                                                           env_info['shape'][1]*FRAME_STACK, 1])
                self.conv1 = conv2d(self.x, 8, 4, 32, CONV1, var_dict=self.dqn_vars)
                self.conv2 = conv2d(self.conv1, 4, 2, 64, CONV2, var_dict=self.dqn_vars)
                self.conv3 = conv2d(self.conv2, 3, 1, 64, CONV3, var_dict=self.dqn_vars)
                conv_shape = self.conv3.get_shape().as_list()
                flatten = [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]]
                self.initial_layers = tf.reshape(self.conv3, flatten)

            # add final hidden layers
            self.hid = fc(self.initial_layers, 512, HIDDEN, var_dict=self.dqn_vars)
            self.q = fc(self.hid, env_info['num_actions'], OUTPUT,
                        var_dict=self.dqn_vars, activation=False)
                          
        # build target network
        self.target_vars = dict()
        with tf.variable_scope(TARGET_SCOPE):
            if env_info['type'] == 'atari':
                self.t_x = tf.placeholder(tf.float32, shape=[None, env_info['shape'][0],
                                                             env_info['shape'][1]*FRAME_STACK, 1])
                self.t_conv1 = conv2d(self.t_x, 8, 4, 32, CONV1, var_dict=self.target_vars)
                self.t_conv2 = conv2d(self.t_conv1, 4, 2, 64, CONV2, var_dict=self.target_vars)
                self.t_conv3 = conv2d(self.t_conv2, 3, 1, 64, CONV3, var_dict=self.target_vars)
                conv_shape = self.t_conv3.get_shape().as_list()
                flatten = [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]]
                self.t_initial_layers = tf.reshape(self.t_conv3, flatten)

            self.t_hid = fc(self.t_initial_layers, 512, HIDDEN, var_dict=self.target_vars)
            self.t_q = fc(self.t_hid, env_info['num_actions'], OUTPUT,
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
            self.action = tf.placeholder(tf.float32, shape=[None, env_info['num_actions']])
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

    def training_predict(self, env, observation):
        # select action according to epsilon-greedy policy
        if random.random() < self.exploration:
            action = env.action_space.sample()
        else:
            action = self.predict(observation)
        self.exploration = max(self.exploration - EXPLORATION_DECAY, FINAL_EXPLORATION)

        return env.action_space.sample() if random.random() < self.exploration else prediction

    def predict(self, observation):
        # push the new observation onto the buffer
        self.observation_buffer.pop(len(self.observation_buffer)-1)
        self.observation_buffer.insert(0, observation)

        # create stacked state for input to dqn
        stacked_state = self.observation_buffer[0]
        for i in range(1, FRAME_STACK):
            stacked_state = np.hstack((stacked_state, self.observation_buffer[i*FRAME_SKIP]))

        return np.argmax(self.sess.run(self.q, feed_dict={self.x: stacked_state}))

    def notify_state_transition(self, state, action, reward, done):
        if done:
            # flush the observation buffer
            for i in range(len(self.observation_buffer)):
                self.observation_buffer[i] = np.zeros(self.observation_buffer[i].shape)

        self.replay_memory.add_state_transition(state, action, reward, done)

    def batch_train(self):
        # sample batch from replay memory
        states, actions, rewards, terminals, newstates = self.replay_memory.sample()
        nonterminals = 1 - terminals

        # update target network weights
        if self.train_iter % TARGET_UPDATE == 0:
            self.sess.run(self.assign_ops)

        # run neural network training step
        self.sess.run(self.optimize, feed_dict={self.x:states, self.t_x:newstates,
                      self.action:actions, self.reward:rewards, self.non_terminal:nonterminals})

        self.train_iter += 1
