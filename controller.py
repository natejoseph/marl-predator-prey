# -*- coding: utf-8 -*-
"""
Created on Tuesday Sep. 26 2023
@author: Nuocheng Yang, MingzheChen
@github: https://github.com/YangNuoCheng, https://github.com/mzchen0 
"""
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras import backend as K

class controller(object):
    def __init__(self, controller_type, n_devices, action_size):
        self.n_devices = n_devices
        self.action_size = action_size
        self.state_shape = (1 + self.n_devices) * 2
        if(controller_type == 'QMIX'):
            self.qmix_hidden_dim = self.n_devices * action_size * action_size
            self.hyper_hidden_dim = 10
            self._build_hyperparameters()
            self.replay = self.QMIXreplay
        elif(controller_type == 'VDN'):
            self.replay = self.VDNreplay
        else:
            self.replay = self.IQLreplay
            
    def IQLreplay(self, agents):
        # print("Applying IQLreplay")
        seeds = random.randint(1, 1e3)
        for agent in agents:
            batch = agent.memory.sample(agent.batch_size, seeds)
            x, y = agent.find_targets_uer(batch)
            agent.brain.model.fit(x, y, batch_size=len(x), sample_weight = None, epochs = 1, verbose = 0)
    
    def VDNreplay(self, agents):
        # print("Applying VDNreplay")
        y_list = []
        y_tot_list = []
        
        seeds = random.randint(1, 1e3)
        for agent in agents:
            batch = agent.memory.sample(agent.batch_size, seeds)
            x, y = agent.find_targets_uer(batch)
            y_list.append(y)
        
        for i, x_i in enumerate(x):
            # *** Design y_tot ***
            y_tot = np.sum(np.array(y_list)[:, i, :], axis = 0) 
            # *** Design y_tot ***
            y_tot_list.append(y_tot)
            
        y_tot_list = np.array(y_tot_list).reshape(-1, self.action_size)
        for i, agent in enumerate(agents):
            agent.brain.model.fit(x, y_tot_list, batch_size=len(x), sample_weight = None, epochs = 1, verbose = 0)
    
    def QMIXreplay(self, agents):
        # print("Applying QMIXreplay")
        y_list = []
        y_tot_list = []
        
        seeds = random.randint(1, 1e3)
        for agent in agents:
            batch = agent.memory.sample(agent.batch_size, seeds)
            x, y = agent.find_targets_uer(batch)
            # x: (1L, 8L), y: (1L, 5L)
            y_list.append(y)
        
        split_vectors = [np.array(y_list)[:, i, :].reshape(1, -1) for i in range(np.array(y_list).shape[1])]
        for i, x_i in enumerate(x):
            self._build_model(np.array([x_i]))
            y_tot = self.Qmixer_w1.predict(np.array(split_vectors[i]))
            y_tot_list.append(y_tot)
                
        y_tot_list = np.array(y_tot_list).reshape(-1, self.action_size)
        for i, agent in enumerate(agents):
            agent.brain.model.fit(x, y_tot_list, batch_size=len(x), sample_weight = None, epochs = 1, verbose = 0)
            
    def positive_init(self, shape, dtype=None):
        random_values = K.random_normal(shape, dtype=dtype)
        random_values = K.abs(random_values)
        return random_values

    def _build_hyperparameters(self):
        self.hyper_w1 = Sequential()
        self.hyper_w1.add(Dense(units=self.hyper_hidden_dim, activation='relu', kernel_initializer=self.positive_init, input_shape=(self.state_shape,)))
        self.hyper_w1.add(Dense(units=self.qmix_hidden_dim, activation='relu', kernel_initializer=self.positive_init, input_shape=(self.qmix_hidden_dim,)))
        
    def _build_model(self, x):
        Qmixer_w1_params = self.hyper_w1.predict(x)
        self.Qmixer_w1 = Sequential()
        self.Qmixer_w1.add(Dense(units = self.action_size, activation='relu', use_bias=False, input_shape=(self.n_devices * self.action_size,)))
        
        self.Qmixer_w1.set_weights(Qmixer_w1_params.reshape([1, self.n_devices * self.action_size, self.action_size]))