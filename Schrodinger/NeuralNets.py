#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:41:40 2021

@author: FMagnani

"""

import tensorflow as tf
from tensorflow.keras.layers import Dense


#%%

class neural_net(tf.keras.Model):
    
    def __init__(self, ub, lb, hidden_dim=40):
        super(neural_net, self).__init__()

        self.lb = lb
        self.ub = ub

        self.in_dim = 3
        self.hidden_dim = hidden_dim
        self.out_dim = 2
        
        self.hidden_1 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")

        self.hidden_2 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")

        self.hidden_3 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")

        self.hidden_4 = Dense(self.hidden_dim, activation='tanh',
                    bias_initializer="zeros")


        self.last_layer = Dense(self.out_dim,
                    bias_initializer="zeros")

    def call(self, X):
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

        H = self.hidden_1(H)
        H = self.hidden_2(H)
        H = self.hidden_3(H)
        H = self.hidden_4(H)
        
        H = self.last_layer(H)
    
        u = H[:,0]
        v = H[:,1]
    
        return u, v



