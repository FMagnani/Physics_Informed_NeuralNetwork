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
        x1 = np.vstack((lb,ub))
        self.x1 = tf.convert_to_tensor(x1)
        self.u0 = u0
        
        self.dt = dt
        self.q = max(q,1)
        
        # Load IRK weights
        tmp = np.float64(np.loadtxt('../Utils/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
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


    def net_U0(self, x):
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            U1 = self.model(x)
            U = U1[:,:-1]
            
            U_x = tape.gradient(U, x)
        
        U_xx = tape.gradient(U_x, x)
        
        del tape
                
        F = 5.0*U - 5.0*U**3 + 0.0001*U_xx
        U0 = U1 - self.dt*tf.matmul(F, self.IRK_weights.T)
        
        return U0
    
    
    def net_U1(self, x):

        with tf.GradientTape() as tape:
            tape.watch(x)
            U1 = self.model(x)

        U1_x = tape.gradient(U1, x)  

        return U1, U1_x # N x (q+1)


    def predict(self, x):
        
        U1_pred = self.model(x)
        
        return U1_pred







