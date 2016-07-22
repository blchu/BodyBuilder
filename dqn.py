import tensorflow as tf

from ops import conv2d, fc, mse
from replay_memory import ReplayMemory

FRAME_STACK = 4
FRAME_SKIP = 4

REPLAY_MEMORY_CAPACITY = 100000  # one hundred thousand

# hyperparameters
GAMMA = 0.9

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

class DQN():

    def __init__(self, env_info):
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)
        self.sess = tf.Session()

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
            self.action = tf.placeholder(tf.float32, shape=[None, env_info['num_actions'])
            # reward
            self.reward = tf.placeholder(tf.float32, shape=[None, 1])
            # terminal state
            self.terminal = tf.placeholder(tf.float32, shape=[None, 1])

            self.target = tf.add(self.reward, tf.mul(gamma, tf.mul(self.terminal,
                          tf.reduce_sum(tf.mul(self.action, self.t_q), 1, True))))
            self.predict = tf.reduce_sum(tf.mul(self.action, self.q), 1, True)
            self.error = mse(self.predict, self.target)

        # initialize variables
        self.sess.run(tf.initialize_all_variables())
