#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:22:26 2021

@author: FMagnani

"""

import sys
sys.path.insert(0, '../Utils/')

import tensorflow as tf
import numpy as np

from NeuralNet import neural_net, PhysicsInformedNN

#%%

class Allen_Cahn_PINN(PhysicsInformedNN):
    
    def __init__(self, x0, u0, layers, dt, lb, ub, q):
        
        super(Allen_Cahn_PINN, self).__init__()
        
        # Network architecture
        self.model = neural_net(ub, lb, layers)        
        
        # Data initialization
        self.x0 = x0
        self.x1 = np.vstack((lb,ub))
        self.u0 = u0
        
        self.dt = dt
        self.q = max(q,1)
        
        # Load IRK weights
        tmp = np.float32(np.loadtxt('../Utils/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
        self.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        self.IRK_times = tmp[q**2+q:]
        
        
        
    # Loss definition
    def loss(self):
        
        u0 = self.u0
        x0 = self.x0
        x1 = self.x1
        
        U0_pred = self.net_U0(x0) # N x (q+1)
        U1_pred, U1_x_pred = self.net_U1(x1) # N1 x (q+1)
        
        y = tf.reduce_sum(tf.square(u0 - U0_pred))
        
        yB = tf.reduce_sum(tf.square(U1_pred[0,:] - U1_pred[1,:])) + \
            tf.reduce_sum(tf.square(U1_x_pred[0,:] - U1_x_pred[1,:])) 
    
        return y + yB

        
















