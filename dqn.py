import os
import shutil

import tensorflow as tf

from ops import conv2d, fc, mse
from replay_memory import ReplayMemory

FRAME_STACK = 4
FRAME_SKIP = 4

REPLAY_MEMORY_CAPACITY = 100000  # one hundred thousand

# hyperparameters
ALPHA = 25e-5    # initial learning rate
GAMMA = 0.9      # discount factor
EPSILON = 1e-2   # numerical stability
DECAY = 0.95     # rmsprop decay
MOMENTUM = 0.95  # rmsprop momentum

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
            self.terminal = tf.placeholder(tf.float32, shape=[None, 1])

            self.target = tf.add(self.reward, tf.mul(GAMMA, tf.mul(self.terminal,
                          tf.reduce_max(self.t_q, 1, True))))
            self.predict = tf.reduce_sum(tf.mul(self.action, self.q), 1, True)
            self.error = mse(self.predict, self.target)
        
        self.optimize = tf.train.RMSPropOptimizer(ALPHA, decay=DECAY, momentum=MOMENTUM,
                        epsilon=EPSILON).minimize(self.error, var_list=self.dqn_vars.values())

        # initialize variables
        self.sess.run(tf.initialize_all_variables())
        
        # write out the graph for tensorboard
        if os.path.isdir(TENSORBOARD_GRAPH_DIR):
            shutil.rmtree(TENSORBOARD_GRAPH_DIR)
        self.writer = tf.train.SummaryWriter(TENSORBOARD_GRAPH_DIR, self.sess.graph)
