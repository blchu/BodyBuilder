import os
import pickle
import random
import shutil
import time

import numpy as np
import tensorflow as tf

from enums import EnvTypes
from ops import conv2d, fc, mse
from replay_memory import ReplayMemory
from utils import rgb_to_luminance, downscale


REPLAY_MEMORY_CAPACITY = 50000  # fifty thousand

# hyperparameters
ALPHA = 1e-4              # initial learning rate
GAMMA = 0.99              # discount factor
EPSILON = 1e-2            # numerical stability
TAU = 0.001               # target network weight transfer decay
RMS_DECAY = 0.95          # rmsprop decay
MOMENTUM = 0.95           # rmsprop momentum
FINAL_EXPLORATION = 0.1   # final exploration rate
EXPLORATION_DECAY = 1e-6  # linear decay of exploration
BATCH_SIZE = 64           # size of training batch
FRAME_STACK = 4           # number of previous frames in a state
FRAME_SKIP = 4            # step size for previous frames

# scope names
DQN_SCOPE = 'dqn'
TARGET_SCOPE = 'target'
TRANSFER_SCOPE = 'transfer'
EVALUATION_SCOPE = 'evaluator'

# layer names
CONV1 = 'conv1'
CONV2 = 'conv2'
CONV3 = 'conv3'
FC = 'fully_connected'
HIDDEN = 'hidden'
OUTPUT = 'output'

TENSORBOARD_GRAPH_DIR = "/tmp/dqn"
SUMMARY_PERIOD = 25
SAVE_CHECKPOINT_PERIOD = 50000

GPU_MEMORY_FRACTION = 0.3

class DQN():

    def __init__(self, env_type, state_dims, num_actions):
        if env_type == EnvTypes.ATARI:
            state_size = [state_dims[0], state_dims[1]*FRAME_STACK, state_dims[2]]
        elif env_type == EnvTypes.STANDARD:
            state_size = state_dims
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_CAPACITY, state_size)
        self.exploration = 1.0
        self.train_iter = 0
        self.env_type = env_type

        if env_type == EnvTypes.ATARI:
            buffer_size = FRAME_STACK*FRAME_SKIP
            self.observation_buffer = [np.zeros((state_dims[0], state_dims[1], state_dims[2]))
                                       for _ in range(buffer_size)]
        else:
            self.observation_buffer = [np.zeros((state_dims[0]))]

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        self.sess = tf.Session(config=self.config)

        # build q network
        self.dqn_vars = dict()
        with tf.variable_scope(DQN_SCOPE):
            if env_type == EnvTypes.ATARI:
                self.x, self.initial_layers = self.add_atari_layers(state_dims, self.dqn_vars)
            elif env_type == EnvTypes.STANDARD:
                self.x, self.initial_layers = self.add_standard_layers(state_dims, self.dqn_vars)

            # add final hidden layers
            self.hid = fc(self.initial_layers, 128, HIDDEN, var_dict=self.dqn_vars)
            self.q = fc(self.hid, num_actions, OUTPUT,
                        var_dict=self.dqn_vars, activation=False)
            
            tf.histogram_summary('q_values', self.q)
                          
        # build target network
        self.target_vars = dict()
        with tf.variable_scope(TARGET_SCOPE):
            if env_type == EnvTypes.ATARI:
                self.t_x, self.t_initial_layers = self.add_atari_layers(state_dims,
                                                                        self.target_vars)
            elif env_type == EnvTypes.STANDARD:
                self.t_x, self.t_initial_layers = self.add_standard_layers(state_dims,
                                                                           self.target_vars)

            self.t_hid = fc(self.t_initial_layers, 128, HIDDEN, var_dict=self.target_vars)
            self.t_q = fc(self.t_hid, num_actions, OUTPUT,
                          var_dict=self.target_vars, activation=False)

            tf.histogram_summary('target_q_values', self.t_q)

        # add weight transfer operations from primary dqn network to target network
        self.assign_ops = []
        with tf.variable_scope(TRANSFER_SCOPE):
            for variable in self.dqn_vars.keys():
                target_variable = TARGET_SCOPE + variable[len(DQN_SCOPE):]
                decay = tf.mul(1 - TAU, self.target_vars[target_variable])
                update = tf.mul(TAU, self.dqn_vars[variable])
                new_target_weight = tf.add(decay, update)
                target_assign = self.target_vars[target_variable].assign(new_target_weight)
                self.assign_ops.append(target_assign)

        # build dqn evaluation
        with tf.variable_scope(EVALUATION_SCOPE):
            # one-hot action selection
            self.action = tf.placeholder(tf.int32, shape=[None])
            self.action_one_hot = tf.one_hot(self.action, num_actions)
            # reward
            self.reward = tf.placeholder(tf.float32, shape=[None, 1])
            # terminal state
            self.nonterminal = tf.placeholder(tf.float32, shape=[None, 1])

            self.target = tf.add(self.reward, tf.mul(GAMMA, tf.mul(self.nonterminal,
                          tf.reduce_max(self.t_q, 1, True))))
            self.predict = tf.reduce_sum(tf.mul(self.action_one_hot, self.q), 1, True)
            self.error = tf.reduce_mean(mse(self.predict, self.target))

            tf.scalar_summary('error', self.error)
        
        val_print = tf.Print(self.error, [self.predict, self.target])
        self.optimize = tf.train.RMSPropOptimizer(ALPHA, decay=RMS_DECAY, momentum=MOMENTUM,
                        epsilon=EPSILON).minimize(self.error, var_list=self.dqn_vars.values())

        # write out the graph and summaries for tensorboard
        self.summaries = tf.merge_all_summaries()
        if os.path.isdir(TENSORBOARD_GRAPH_DIR):
            shutil.rmtree(TENSORBOARD_GRAPH_DIR)
        self.writer = tf.train.SummaryWriter(TENSORBOARD_GRAPH_DIR, self.sess.graph)

        # initialize variables
        self.sess.run(tf.initialize_all_variables())

        # create saver
        self.saver = tf.train.Saver()

    def add_atari_layers(self, dims, var_dict):
        x = tf.placeholder(tf.float32, shape=[None, dims[0], dims[1]*FRAME_STACK, 1])
        conv1 = conv2d(x, 8, 4, 32, CONV1, var_dict=var_dict)
        conv2 = conv2d(conv1, 4, 2, 64, CONV2, var_dict=var_dict)
        conv3 = conv2d(conv2, 3, 1, 64, CONV3, var_dict=var_dict)
        conv_shape = conv3.get_shape().as_list()
        flatten = [-1, conv_shape[1]*conv_shape[2]*conv_shape[3]]
        return x, tf.reshape(conv3, flatten)

    def add_standard_layers(self, dims, var_dict):
        x = tf.placeholder(tf.float32, shape=[None, dims[0]])
        fc1 = fc(x, 256, FC, var_dict=var_dict)
        return x, fc1
        
    def process_observation(self, observation):
        if self.env_type == EnvTypes.ATARI:
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
        if self.env_type == EnvTypes.ATARI:
            state = self._get_stacked_state()
        else:
            state = self.observation_buffer[0]
        state = np.expand_dims(state, axis=0)
        return np.argmax(self.sess.run(self.q, feed_dict={self.x: state}))

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
        if self.env_type == EnvTypes.ATARI:
            state = self._get_stacked_state()
        else:
            state = self.observation_buffer[0]
        self.replay_memory.add_state_transition(state, action, reward, done)
        if done:
            # flush the observation buffer
            for i in range(len(self.observation_buffer)):
                self.observation_buffer[i] = np.zeros(self.observation_buffer[i].shape)

    def batch_train(self, save_dir):
        # sample batch from replay memory
        state, action, reward, terminal, newstate = self.replay_memory.sample(BATCH_SIZE)
        reward = np.expand_dims(reward, axis=1)
        terminal = np.expand_dims(terminal, axis=1)
        nonterminal = 1 - terminal

        # update target network weights
        self.sess.run(self.assign_ops)

        # run neural network training step
        if self.train_iter % SUMMARY_PERIOD == 0:
            summary, _ = self.sess.run([self.summaries, self.optimize], feed_dict={self.x:state,
                                       self.t_x:newstate, self.action:action,
                                       self.reward:reward, self.nonterminal:nonterminal})
            self.writer.add_summary(summary, self.train_iter)
        else:
            self.sess.run(self.optimize, feed_dict={self.x:state, self.t_x:newstate,
                          self.action:action, self.reward:reward, self.nonterminal:nonterminal})

        # save the dqn
        if save_dir is not None and self.train_iter % SAVE_CHECKPOINT_PERIOD == 0:
            self.save_algorithm(save_dir)

        self.train_iter += 1

    def save_algorithm(self, save_dir):
        # create directory tree for saving the algorithm
        checkpoint_dir = save_dir + "/save_{}".format(self.train_iter)
        os.mkdir(checkpoint_dir)
        model_file = checkpoint_dir + "/model.ckpt"

        print("Saving algorithm to {}".format(checkpoint_dir))
        t = time.time()
        self.saver.save(self.sess, model_file)
        print("Completed saving in {} seconds".format(time.time() - t))

    def restore_algorithm(self, restore_dir):
        self.train_iter = int(restore_dir[restore_dir.rfind("save_") + len("save_"):])
        self.saver.restore(self.sess, restore_dir + "/model.ckpt")
