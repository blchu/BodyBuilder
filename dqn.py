import tensorflow as tf

from ops import conv2d, fc, mse
from replay_memory import ReplayMemory

FRAME_STACK = 4
FRAME_SKIP = 4

REPLAY_MEMORY_CAPACITY = 1000000  # one million

class DQN():

    def __init__(self, replay_memory, env_info):
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)
        self.sess = tf.InteractiveSession()

        # build network
        self.dqn_var_dict = dict()
        with tf.variable_scope('dqn'):
            if env_info['type'] == 'atari':
                self.x = tf.placeholder(tf.float32, shape=[None, env_info['shape'][0],
                                                           env_info['shape'][1]*FRAME_STACK, 1])
                initial_layers = self._add_atari_conv_layers()
                flatten = [-1, int(env_info['shape'][0]*env_info['shape'][1]*FRAME_STACK/(4*2*2))]
                initial_layers = tf.reshape(initial_layers, flatten)

            # add final hidden layers
            self.hid = fc(initial_layers, 512, 'hiddenlayer')
            self.out = fc(self.hid, env_info['num_actions'], 'outputlayer', activation=False)

    def _add_atari_conv_layers(self):
        self.conv1 = conv2d(self.x, 8, 4, 32, 'convlayer1', var_dict=self.dqn_var_dict)
        self.conv2 = conv2d(self.conv1, 4, 2, 64, 'convlayer2', var_dict=self.dqn_var_dict)
        self.conv3 = conv2d(self.conv2, 3, 1, 64, 'convlayer3', var_dict=self.dqn_var_dict)
        return self.conv3
