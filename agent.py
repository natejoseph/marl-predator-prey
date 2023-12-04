# -*- coding: utf-8 -*-
"""
Created on Tuesday Sep. 26 2023
@author: Nuocheng Yang, MingzheChen
@github: https://github.com/YangNuoCheng, https://github.com/mzchen0 
"""

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras import backend as K
from keras.optimizers import *
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Concatenate, BatchNormalization, Activation

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0
HUBER_LOSS_DELTA = 1.0

def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)

class Brain(object):
    def __init__(self, state_size, action_size, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.optimizer_model = arguments['optimizer']
        self.model = self.build_model()
        self.model_ = self.build_model()
#nina
    def build_model(self):

        x = Input(shape=(self.state_size,))
        # design a neural network model Q(s,a)
        # Hidden layers
        hidden_layer = Dense(self.num_nodes, activation="relu")(x)
        hidden_layer = Dense(128, activation="relu")(hidden_layer)
        hidden_layer = Dense(64, activation="relu")(hidden_layer)
        hidden_layer = Dense(32, activation="relu")(hidden_layer)
        hidden_layer = Dense(16, activation="relu")(hidden_layer)

        #Output layer
        z = Dense(self.action_size, activation="relu")(hidden_layer)

        model = Model(inputs=x, outputs=z)
        
        if self.optimizer_model == 'Adam':
            optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(lr=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')

        model.compile(loss=huber_loss, optimizer=optimizer)
        
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model.load_weights(self.weight_backup)

        return model

    def predict(self, state, target=False):
        if target:
            return self.model_.predict(state)
        else:
            return self.model.predict(state)

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1,self.state_size), target=target).flatten()

    def save_model(self):
        self.model.save(self.weight_backup)
        
class Agent(object):
    
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, state_size, action_size, bee_index, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.bee_index = bee_index
        self.learning_rate = arguments['learning_rate']
        self.gamma = arguments['gamma']
        self.brain = Brain(self.state_size, self.action_size, brain_name, arguments)
        self.memory = UER(arguments['memory_capacity'])
        self.target_type = arguments['target_type']
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0
        self.test = arguments['test']
        if self.test:
            self.epsilon = MIN_EPSILON

    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict_one_sample(state))

    def find_targets_uer(self, batch):
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1][self.bee_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

    def observe(self, sample):
        self.memory.remember(sample)

    def decay_epsilon(self):
        # slowly decrease Epsilon based on our experience
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON
                
class UER(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def remember(self, sample):
        self.memory.append(sample)

    def sample(self, n, seeds = 1):
        random.seed(seeds)
        n = min(n, len(self.memory))
        sample_batch = random.sample(self.memory, n)

        return sample_batch